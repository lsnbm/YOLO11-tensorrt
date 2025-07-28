import os
import pickle
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Union

import onnx
import tensorrt as trt
import torch

# 建议的最佳实践，避免在不需要时过早加载CUDA上下文
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


class EngineBuilder:
    """
    Builds a TensorRT engine from an ONNX or a custom API definition.
    """
    seg = False  # Placeholder for segmentation models, not used in this context

    def __init__(
            self,
            checkpoint: Union[str, Path],
            device: Optional[Union[str, int, torch.device]] = None) -> None:
        """
        Initializes the EngineBuilder.

        Args:
            checkpoint: Path to the ONNX model or a .pkl file for API-based build.
            device: The device to use for building the engine (e.g., 'cuda:0').
        """
        checkpoint = Path(checkpoint) if isinstance(checkpoint,
                                                    str) else checkpoint
        assert checkpoint.exists() and checkpoint.suffix in ('.onnx', '.pkl')
        self.api = checkpoint.suffix == '.pkl'
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')

        self.checkpoint = checkpoint
        self.device = device if device is not None else torch.device('cuda:0')

    def __build_engine(self,
                       fp16: bool = True,
                       input_shape: Union[List, Tuple] = (1, 3, 640, 640),
                       iou_thres: float = 0.65,
                       conf_thres: float = 0.25,
                       topk: int = 100,
                       with_profiling: bool = True) -> None:
        """
        The private method that orchestrates the engine build process.
        """
        # 1. 初始化 Logger, Builder, 和 Network
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace='')
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # 2. 创建 Builder Config
        config = self.builder.create_builder_config()

        # [API UPDATE] 'max_workspace_size' is deprecated.
        # Use 'set_memory_pool_limit' with MemoryPoolType.WORKSPACE instead.
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     torch.cuda.get_device_properties(
                                         self.device).total_memory // 2)

        # 3. 根据输入类型构建网络 (ONNX or API)
        if self.api:
            # This path is for building from a custom API definition (not updated here)
            self.build_from_api(fp16, input_shape, iou_thres, conf_thres, topk)
        else:
            self.build_from_onnx(iou_thres, conf_thres, topk)

        # 4. 设置精度
        if fp16 and self.builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        self.weight = self.checkpoint.with_suffix('.engine')

        # 5. 设置 Profiling
        if with_profiling:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # [API UPDATE] builder.build_engine is deprecated.
        # Use builder.build_serialized_network instead.
        serialized_engine = self.builder.build_serialized_network(self.network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build the TensorRT engine. Check the logs for errors.")
        
        # 6. 保存引擎文件
        self.weight.write_bytes(serialized_engine)
        self.logger.log(
            trt.Logger.WARNING, f'Build TensorRT engine finish.\n'
            f'Save in {str(self.weight.absolute())}')

    def build(self,
              fp16: bool = True,
              input_shape: Union[List, Tuple] = (1, 3, 640, 640),
              iou_thres: float = 0.65,
              conf_thres: float = 0.25,
              topk: int = 100,
              with_profiling=True) -> None:
        """
        Public method to start the build process.
        """
        self.__build_engine(fp16, input_shape, iou_thres, conf_thres, topk,
                            with_profiling)

    def build_from_onnx(self,
                        iou_thres: float = 0.65,
                        conf_thres: float = 0.25,
                        topk: int = 100):
        """
        Builds the network from an ONNX file.
        """
        parser = trt.OnnxParser(self.network, self.logger)
        onnx_model = onnx.load(str(self.checkpoint))

        # This part manually modifies the ONNX graph. It's brittle but works for this specific model.
        if not self.seg:
            try:
                # Find the EfficientNMS_TRT node and update its attributes
                nms_node = next(node for node in onnx_model.graph.node if node.op_type == 'EfficientNMS_TRT')
                for attr in nms_node.attribute:
                    if attr.name == 'max_output_boxes_per_class':
                        attr.i = topk
                    elif attr.name == 'score_threshold':
                        attr.f = conf_thres
                    elif attr.name == 'iou_threshold':
                        attr.f = iou_thres
            except StopIteration:
                self.logger.log(trt.Logger.WARNING, "Could not find EfficientNMS_TRT node to modify. Using ONNX-defined values.")


        if not parser.parse(onnx_model.SerializeToString()):
            for error in range(parser.num_errors):
                self.logger.log(trt.Logger.ERROR, parser.get_error(error))
            raise RuntimeError(
                f'Failed to load ONNX file: {str(self.checkpoint)}')

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        for inp in inputs:
            self.logger.log(trt.Logger.WARNING, f'Input "{inp.name}" with shape: {inp.shape} dtype: {inp.dtype}')
        for out in outputs:
            self.logger.log(trt.Logger.WARNING, f'Output "{out.name}" with shape: {out.shape} dtype: {out.dtype}')

    def build_from_api(
        self,
        fp16: bool = True,
        input_shape: Union[List, Tuple] = (1, 3, 640, 640),
        iou_thres: float = 0.65,
        conf_thres: float = 0.25,
        topk: int = 100,
    ):
        """
        Placeholder for building from a custom API definition.
        NOTE: This part is highly specific and would require its own update if used.
        """
        self.logger.log(trt.Logger.ERROR, "Building from API is not fully updated and is not recommended.")
        # The original API build code would go here. It's omitted for clarity as it depends on an unprovided '.api' module.
        pass


class TRTModule(torch.nn.Module):
    """
    A PyTorch nn.Module wrapper for a TensorRT engine for easy inference.
    """
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        """Initializes the TensorRT runtime, engine, and execution context."""
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        runtime = trt.Runtime(logger)
        model = runtime.deserialize_cuda_engine(self.weight.read_bytes())
        self.context = model.create_execution_context()
        self.model = model

    def __init_bindings(self) -> None:
        """Initializes input and output bindings for the engine using modern APIs."""
        idynamic = odynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []

        # Use the modern API to iterate over I/O tensors
        for i in range(self.model.num_io_tensors):
            name = self.model.get_tensor_name(i)
            dtype = self.dtypeMapping[self.model.get_tensor_dtype(name)]
            shape = self.model.get_tensor_shape(name)
            
            if self.model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if -1 in shape:
                    idynamic = True
                inp_info.append(Tensor(name, dtype, shape))
            else: # Output tensor
                if -1 in shape:
                    odynamic = True
                out_info.append(Tensor(name, dtype, shape))

        self.num_inputs = len(inp_info)
        self.num_outputs = len(out_info)
        self.input_names = [info.name for info in inp_info]
        self.output_names = [info.name for info in out_info]
        self.idx = list(range(self.num_outputs)) # Default output order

        # Pre-allocate output tensors if shapes are static
        if not odynamic:
            self.output_tensor = [
                torch.empty(info.shape, dtype=info.dtype, device=self.device)
                for info in out_info
            ]
        self.idynamic = idynamic
        self.odynamic = odynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        """Sets a profiler for the execution context."""
        self.context.profiler = profiler if profiler is not None else trt.Profiler()

    def set_desired(self, desired: Optional[Union[List, Tuple]]):
        """Sets the desired order of outputs."""
        if isinstance(desired, (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> Union[Tuple, torch.Tensor]:
        """
        The forward pass using the modern execute_async_v3 API.
        """
        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[torch.Tensor] = [i.contiguous() for i in inputs]

        # Set input tensor addresses and shapes (if dynamic)
        for i, name in enumerate(self.input_names):
            if self.idynamic:
                self.context.set_input_shape(name, tuple(contiguous_inputs[i].shape))
            self.context.set_tensor_address(name, contiguous_inputs[i].data_ptr())

        # Prepare output tensors and set their addresses
        outputs: List[torch.Tensor] = []
        for i, name in enumerate(self.output_names):
            if self.odynamic:
                shape = self.context.get_tensor_shape(name)
                output = torch.empty(size=shape, dtype=self.out_info[i].dtype, device=self.device)
            else:
                output = self.output_tensor[i]
            self.context.set_tensor_address(name, output.data_ptr())
            outputs.append(output)
        
        # Execute asynchronously using the modern v3 API
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return tuple(outputs[i] for i in self.idx) if len(outputs) > 1 else outputs[0]


class TRTProfilerV1(trt.IProfiler):
    """A simple profiler to measure layer-wise execution time in microseconds."""
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.total_runtime = 0.0
        self.recorder = defaultdict(float)

    def report_layer_time(self, layer_name: str, ms: float):
        self.total_runtime += ms * 1000
        self.recorder[layer_name] += ms * 1000

    def report(self):
        f = '\t%40s\t\t\t\t%10.4f'
        print('\t%40s\t\t\t\t%10s' % ('layername', 'cost(us)'))
        for name, cost in sorted(self.recorder.items(), key=lambda x: -x[1]):
            print(f % (name if len(name) < 40 else name[:35] + ' ' + '*' * 4, cost))
        print(f'\nTotal Inference Time: {self.total_runtime:.4f}(us)')


class TRTProfilerV0(trt.IProfiler):
    """A simple profiler to measure layer-wise execution time in milliseconds."""
    def __init__(self):
        trt.IProfiler.__init__(self)

    def report_layer_time(self, layer_name: str, ms: float):
        f = '\t%40s\t\t\t\t%10.4fms'
        print(f % (layer_name if len(layer_name) < 40 else layer_name[:35] + ' ' + '*' * 4, ms))