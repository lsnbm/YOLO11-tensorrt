#include "NvInferPlugin.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <chrono>
#include <windows.h>
#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
#include "Take a screenshot.hpp"

// --- ���ӿ� (��������Ҫ����������) ---
#pragma comment(lib, "Shcore.lib")
#pragma comment(lib, "nvinfer_10.lib")
#pragma comment(lib, "nvinfer_dispatch_10.lib")
#pragma comment(lib, "nvinfer_lean_10.lib")
#pragma comment(lib, "nvinfer_plugin_10.lib")
#pragma comment(lib, "nvinfer_vc_plugin_10.lib")
#pragma comment(lib, "nvonnxparser_10.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cublasLt.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudadevrt.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cudart_static.lib")
#pragma comment(lib, "cudnn.lib")
#pragma comment(lib, "cudnn_adv.lib")
#pragma comment(lib, "cudnn_adv64_9.lib")
#pragma comment(lib, "cudnn_cnn.lib")
#pragma comment(lib, "cudnn_cnn64_9.lib")
#pragma comment(lib, "cudnn_engines_precompiled.lib")
#pragma comment(lib, "cudnn_engines_precompiled64_9.lib")
#pragma comment(lib, "cudnn_engines_runtime_compiled.lib")
#pragma comment(lib, "cudnn_engines_runtime_compiled64_9.lib")
#pragma comment(lib, "cudnn_graph.lib")
#pragma comment(lib, "cudnn_graph64_9.lib")
#pragma comment(lib, "cudnn_heuristic.lib")
#pragma comment(lib, "cudnn_heuristic64_9.lib")
#pragma comment(lib, "cudnn_ops.lib")
#pragma comment(lib, "cudnn_ops64_9.lib")
#pragma comment(lib, "cudnn64_9.lib")
#pragma comment(lib, "cufft.lib")
#pragma comment(lib, "cufftw.lib")
#pragma comment(lib, "cufilt.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cusolver.lib")
#pragma comment(lib, "cusolverMg.lib")
#pragma comment(lib, "cusparse.lib")
#pragma comment(lib, "nppc.lib")
#pragma comment(lib, "nppial.lib")
#pragma comment(lib, "nppicc.lib")
#pragma comment(lib, "nppidei.lib")
#pragma comment(lib, "nppif.lib")
#pragma comment(lib, "nppig.lib")
#pragma comment(lib, "nppim.lib")
#pragma comment(lib, "nppist.lib")
#pragma comment(lib, "nppisu.lib")
#pragma comment(lib, "nppitc.lib")
#pragma comment(lib, "npps.lib")
#pragma comment(lib, "nvblas.lib")
#pragma comment(lib, "nvfatbin.lib")
#pragma comment(lib, "nvfatbin_static.lib")
#pragma comment(lib, "nvJitLink.lib")
#pragma comment(lib, "nvJitLink_static.lib")
#pragma comment(lib, "nvjpeg.lib")
#pragma comment(lib, "nvml.lib")
#pragma comment(lib, "nvptxcompiler_static.lib")
#pragma comment(lib, "nvrtc.lib")
#pragma comment(lib, "nvrtc_static.lib")
#pragma comment(lib, "nvrtc-builtins_static.lib")
#pragma comment(lib, "OpenCL.lib")
#pragma comment(lib, "opencv_world4120 .lib")
// --- ���ӿ���� ---

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            std::cerr << "CUDA Error (" << __FILE__ << ":" << __LINE__ << "): "                                         \
                      << cudaGetErrorString(error_code) << std::endl;                                                  \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// --- ���������ͽṹ�� ---
inline int get_size_by_dims(const nvinfer1::Dims& dims) {
	int size = 1;
	for (int i = 0; i < dims.nbDims; ++i) size *= dims.d[i];
	return size;
}
inline float clamp(float val, float min, float max) {
	return val > min ? (val < max ? val : max) : min;
}

struct Binding { size_t size = 1, dsize = 1; nvinfer1::Dims dims; std::string name; };
struct Object {
	cv::Rect_<float> rect;  // �洢Ŀ��ľ��ο���Ϣ
	int              label; // Ŀ�������ǩ
	float            prob;  // Ŀ������Ŷ�
};
struct PreParam { float ratio = 1.0f, dw = 0.0f, dh = 0.0f, height = 0, width = 0; };

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) std::cerr << msg << std::endl;
	}
};

// --- YOLOv11 �ඨ�� ---
class YOLO11 {
public:
	// ���캯������ʼ���ͼ���ģ��
	explicit YOLO11(const std::string& engine_file_path) {
		runtime = nullptr; engine = nullptr; context = nullptr; stream = nullptr;

		std::ifstream file(engine_file_path, std::ios::binary);
		if (!file.is_open()) throw std::runtime_error("�޷��� engine �ļ�: " + engine_file_path);

		file.seekg(0, std::ios::end);
		size_t size = file.tellg();
		file.seekg(0, std::ios::beg);
		std::vector<char> trtModelStream(size);
		file.read(trtModelStream.data(), size);
		file.close();

		initLibNvInferPlugins(&gLogger, "");
		runtime = nvinfer1::createInferRuntime(gLogger);
		engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
		context = engine->createExecutionContext();
		if (!context) throw std::runtime_error("TensorRT �����ʼ��ʧ�ܡ�");

		CHECK(cudaStreamCreate(&stream));

		for (int i = 0; i < engine->getNbIOTensors(); ++i) {
			Binding binding;
			const char* name = engine->getIOTensorName(i);
			binding.name = name;
			binding.dims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
			binding.size = get_size_by_dims(binding.dims);
			binding.dsize = sizeof(float); // ������������Ϊ float

			if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
				input_bindings.push_back(binding);
			}
			else {
				output_bindings.push_back(binding);
			}
		}
	}

	// �����������ͷ�������Դ
	~YOLO11() {
		if (stream) cudaStreamDestroy(stream);
		for (void* ptr : host_ptrs) CHECK(cudaFreeHost(ptr));
		for (void* ptr : device_ptrs) CHECK(cudaFree(ptr));

		// TensorRT 10 ʹ�� delete �ͷŶ���
		if (context) delete context;
		if (engine) delete engine;
		if (runtime) delete runtime;
	}

	// �߲���ӿ�
	void detect(const cv::Mat& image, std::vector<Object>& objs, long long& inference_ms) {
		copy_from_Mat(image);
		auto start = std::chrono::steady_clock::now();
		infer();
		auto end = std::chrono::steady_clock::now();
		inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		postprocess(objs);
	}

	// �����ڴ�ܵ���Ԥ��
	void make_pipe(bool warmup = true) {
		for (const auto& binding : input_bindings) {
			void* d_ptr;
			CHECK(cudaMalloc(&d_ptr, binding.size * binding.dsize));
			device_ptrs.push_back(d_ptr);
			context->setTensorAddress(binding.name.c_str(), d_ptr);
		}
		for (const auto& binding : output_bindings) {
			void* d_ptr, * h_ptr;
			size_t size = binding.size * binding.dsize;
			CHECK(cudaMalloc(&d_ptr, size));
			CHECK(cudaHostAlloc(&h_ptr, size, 0));
			device_ptrs.push_back(d_ptr);
			host_ptrs.push_back(h_ptr);
			context->setTensorAddress(binding.name.c_str(), d_ptr);
		}

		if (warmup) {
			std::cout << "ģ��Ԥ����..." << std::endl;
			std::vector<char> dummy_input(input_bindings[0].size * input_bindings[0].dsize, 0);
			for (int i = 0; i < 10; i++) {
				CHECK(cudaMemcpy(device_ptrs[0], dummy_input.data(), dummy_input.size(), cudaMemcpyHostToDevice));
				infer();
			}
			std::cout << "Ԥ����ɡ�" << std::endl;
		}
	}

	// ��̬���ƺ���
	static void draw_objects(const cv::Mat& image, const std::vector<Object>& objs, const std::vector<std::string>& CLASS_NAMES, const std::vector<std::vector<unsigned int>>& COLORS) {
		for (const auto& obj : objs) {
			if (obj.label < 0 || static_cast<size_t>(obj.label) >= CLASS_NAMES.size()) continue;
			cv::Scalar color(COLORS[obj.label][2], COLORS[obj.label][1], COLORS[obj.label][0]);
			cv::rectangle(image, obj.rect, color, 2);
			char text[256];
			sprintf_s(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);
			int baseLine = 0;
			cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			int y = static_cast<int>(obj.rect.y) - label_size.height - baseLine;
			cv::Point text_origin(static_cast<int>(obj.rect.x), y < 0 ? static_cast<int>(obj.rect.y) + baseLine + label_size.height : y + label_size.height);
			cv::rectangle(image, cv::Rect(text_origin.x, text_origin.y - label_size.height - baseLine, label_size.width, label_size.height + baseLine), color, -1);
			cv::putText(image, text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 255, 255, 255 }, 1);
		}
	}

private:
	// ͼ��Ԥ����
	void letterbox(const cv::Mat& image, cv::Mat& out, const cv::Size& new_shape) {
		float r = (static_cast<float>(new_shape.height) / image.rows) < (static_cast<float>(new_shape.width) / image.cols)
			? (static_cast<float>(new_shape.height) / image.rows) : (static_cast<float>(new_shape.width) / image.cols);
		int pad_w = static_cast<int>(std::round(image.cols * r));
		int pad_h = static_cast<int>(std::round(image.rows * r));
		cv::Mat tmp;
		cv::resize(image, tmp, cv::Size(pad_w, pad_h));

		pparam.dw = (new_shape.width - pad_w) / 2.0f;
		pparam.dh = (new_shape.height - pad_h) / 2.0f;
		pparam.ratio = 1.0f / r;
		pparam.height = image.rows;
		pparam.width = image.cols;

		int top = static_cast<int>(std::round(pparam.dh - 0.1f));
		int bottom = static_cast<int>(std::round(pparam.dh + 0.1f));
		int left = static_cast<int>(std::round(pparam.dw - 0.1f));
		int right = static_cast<int>(std::round(pparam.dw + 0.1f));

		cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, { 114, 114, 114 });
		cv::dnn::blobFromImage(tmp, out, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
	}

	// �������ݵ��豸
	void copy_from_Mat(const cv::Mat& image) {
		cv::Mat nchw;
		const auto& in_binding = input_bindings[0];
		cv::Size input_shape(in_binding.dims.d[3], in_binding.dims.d[2]);
		letterbox(image, nchw, input_shape);
		context->setInputShape(in_binding.name.c_str(), nvinfer1::Dims{ 4, {1, 3, input_shape.height, input_shape.width} });
		CHECK(cudaMemcpyAsync(device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream));
	}

	// ִ������
	void infer() {
		context->enqueueV3(stream);
		for (size_t i = 0; i < output_bindings.size(); ++i) {
			size_t osize = output_bindings[i].size * output_bindings[i].dsize;
			CHECK(cudaMemcpyAsync(host_ptrs[i], device_ptrs[i + input_bindings.size()], osize, cudaMemcpyDeviceToHost, stream));
		}
		cudaStreamSynchronize(stream);
	}

	// �������
	void postprocess(std::vector<Object>& objs) {
		objs.clear();
		int* num_dets = static_cast<int*>(host_ptrs[0]);
		float* boxes = static_cast<float*>(host_ptrs[1]);
		float* scores = static_cast<float*>(host_ptrs[2]);
		int* labels = static_cast<int*>(host_ptrs[3]);

		for (int i = 0; i < num_dets[0]; ++i) {
			float* ptr = boxes + i * 4;
			Object obj;
			float x0 = (ptr[0] - pparam.dw) * pparam.ratio;
			float y0 = (ptr[1] - pparam.dh) * pparam.ratio;
			float x1 = (ptr[2] - pparam.dw) * pparam.ratio;
			float y1 = (ptr[3] - pparam.dh) * pparam.ratio;
			obj.rect.x = clamp(x0, 0, pparam.width);
			obj.rect.y = clamp(y0, 0, pparam.height);
			obj.rect.width = clamp(x1, 0, pparam.width) - obj.rect.x;
			obj.rect.height = clamp(y1, 0, pparam.height) - obj.rect.y;
			obj.prob = scores[i];
			obj.label = labels[i];
			objs.push_back(obj);
		}
	}

	// ��Ա����
	Logger gLogger;
	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	cudaStream_t stream;

	std::vector<Binding> input_bindings;
	std::vector<Binding> output_bindings;
	std::vector<void*> device_ptrs;
	std::vector<void*> host_ptrs;
	PreParam pparam;
};

// --- ������ɫ���� ---
static const std::vector<std::string> CLASS_NAMES = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
static const std::vector<std::vector<unsigned int>> COLORS = { {100, 114, 189}, {217, 83, 25}, {237, 177, 32}, {126, 47, 142}, {119, 172, 48}, {77, 190, 238}, {162, 20, 47}, {76, 76, 76}, {153, 153, 153}, {255, 0, 0}, {255, 128, 0}, {191, 191, 0}, {0, 255, 0}, {0, 0, 255}, {170, 0, 255}, {85, 85, 0}, {85, 170, 0}, {85, 255, 0}, {170, 85, 0}, {170, 170, 0}, {170, 255, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {0, 85, 128}, {0, 170, 128}, {0, 255, 128}, {85, 0, 128}, {85, 85, 128}, {85, 170, 128}, {85, 255, 128}, {170, 0, 128}, {170, 85, 128}, {170, 170, 128}, {170, 255, 128}, {255, 0, 128}, {255, 85, 128}, {255, 170, 128}, {255, 255, 128}, {0, 85, 255}, {0, 170, 255}, {0, 255, 255}, {85, 0, 255}, {85, 85, 255}, {85, 170, 255}, {85, 255, 255}, {170, 0, 255}, {170, 85, 255}, {170, 170, 255}, {170, 255, 255}, {255, 0, 255}, {255, 85, 255}, {255, 170, 255}, {85, 0, 0}, {128, 0, 0}, {170, 0, 0}, {212, 0, 0}, {255, 0, 0}, {0, 43, 0}, {0, 85, 0}, {0, 128, 0}, {0, 170, 0}, {0, 212, 0}, {0, 255, 0}, {0, 0, 43}, {0, 0, 85}, {0, 0, 128}, {0, 0, 170}, {0, 0, 212}, {0, 0, 255}, {0, 0, 0}, {36, 36, 36}, {73, 73, 73}, {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189}, {80, 183, 189}, {128, 128, 0} };



int main() {
	// // ��ʼ�� CUDA �� YOLO ģ��
	cudaSetDevice(0);
	YOLO11* yolo = new YOLO11("trtexec-yolo11n.engine");
	yolo->make_pipe(true);

	// // ����ͼ������صı���
	cv::Mat image;
	std::vector<Object> objs;
	long long inference_ms = 0;
	int mode_selection = 0;   

	// --- ģʽѡ�� ---
	std::cout << "   1: ����Ƶ�ļ�������ͷ���м��" << std::endl;
	std::cout << "   2: ��ʵʱ��Ļ������м��" << std::endl;
	std::cout << "������ѡ�� (1 �� 2): ";
	std::cin >> mode_selection; 

	try {
		// --- ����ѡ���ģʽִ�� ---
		switch (mode_selection) { 
		case 1: {
			// --- ģʽ 1: ��Ƶ�ļ�������ͷ ---
			std::cout << "��ѡ��ģʽ 1: ��Ƶ/����ͷ���" << std::endl;
			cv::VideoCapture cap;
			// // ���Դ���Ƶ�ļ������ʧ�����Դ�����ͷ
			if (!cap.open("../shiping.mp4")) {
				std::cout << "�޷�����Ƶ�ļ������Դ�Ĭ������ͷ..." << std::endl;
				if (!cap.open(0)) {
					std::cerr << "����: ��Ƶ������ͷ���޷��򿪣�" << std::endl;
					break;
				}
			}

			while (cap.read(image)) {

				if (image.empty()) break;
				// // �� 'Q' ���˳�ѭ��
				if (GetAsyncKeyState('Q') & 0x8000) {break;}


				// // ִ�м��ͻ��ƣ�inference_ms �����ﱻ��ȷ�����ڷ��غ�ʱ
				yolo->detect(image, objs, inference_ms);
				YOLO11::draw_objects(image, objs, CLASS_NAMES, COLORS);

				printf("��⵽ %zu ��Ŀ�꣬��ʱ %lld ���롣\n", objs.size(), inference_ms);
				// // ���� OpenCV ��ʾ�ӿ�
				cv::imshow("OpenCV ����", image);


				cv::waitKey(1);//������imshow���ʹ��
			}
			break;
		}

		case 2: {
			// --- ģʽ 2: ʵʱ��Ļ���� ---
			std::cout << "��ѡ��ģʽ 2: ʵʱ��Ļ��⡣�� 'Q' ���˳���" << std::endl;

			// // ��ʼ����Ļ������
			ScreenCapturer capturer;
			if (!capturer.is_valid()) {
				std::cerr << "����: �޷���ʼ����Ļ��������" << std::endl;
				return -1;
			}

			while (true) {
				// // �� 'Q' ���˳�ѭ��
				if (GetAsyncKeyState('Q') & 0x8000) {break;}

				// // ����һ֡��Ļ
				image = capturer.capture_frame();

				if (image.empty()) {continue;}

				 // �������4ͨ��BGRAͼ��ת��ΪYOLOģ����Ҫ��3ͨ��BGRͼ��
				cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

				 // ִ�м��ͻ���
				yolo->detect(image, objs, inference_ms);
				YOLO11::draw_objects(image, objs, CLASS_NAMES, COLORS);
				printf("��⵽ %zu ��Ŀ�꣬��ʱ %lld ���롣\n", objs.size(), inference_ms);

	
				show_image_in_win32_window("Win32 ���� - ʵʱ��Ļ���", image);
			}
			break;
		}

		default: {
			// --- ��Чѡ�� ---
			std::cerr << "������Ч�������� 1 �� 2��" << std::endl;
			break;
		}
		}
	}
	catch (const std::exception& e) {
		std::cerr << "�������쳣: " << e.what() << std::endl;
	}

	// // ������Դ
	delete yolo;
	cv::destroyAllWindows(); // // �ر������� OpenCV �����Ĵ���

	system("pause"); // // ���˳�ǰ��ͣ������鿴����̨���
	return 0;
}