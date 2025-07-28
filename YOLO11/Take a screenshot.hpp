#pragma once // 防止头文件被重复包含

// --- 包含的头文件 ---
#include <windows.h>      // // Win32 API 核心头文件
#include <d3d11.h>        // // DirectX 11 核心头文件
#include <dxgi1_2.h>      // // DXGI (DirectX Graphics Infrastructure) 头文件，用于桌面复制
#include <wrl/client.h>   // // ComPtr 智能指针，用于自动管理 COM 对象生命周期
#include <opencv2/core.hpp> // // OpenCV 核心库，仅使用 cv::Mat 结构

#include <vector>         // // C++ 标准库，动态数组
#include <string>         // // C++ 标准库，字符串处理
#include <map>            // // C++ 标准库，用于将窗口句柄映射到图像
#include <iostream>       // // C++ 标准库，用于控制台输出（例如错误信息）

// --- 链接必要的库 (避免在项目设置中手动添加) ---
#pragma comment(lib, "d3d11.lib")    // // 链接 DirectX 11 库
#pragma comment(lib, "dxgi.lib")     // // 链接 DXGI 库
#pragma comment(lib, "user32.lib")   // // 链接 Windows 用户界面库 (窗口管理)
#pragma comment(lib, "gdi32.lib")    // // 链接 Windows GDI 库 (图形设备接口，用于绘图)

// // 使用 ComPtr 智能指针的命名空间
using Microsoft::WRL::ComPtr;


class ScreenCapturer {
public:
    // // 构造函数：在创建对象时自动初始化屏幕捕获器
    ScreenCapturer() {
        HRESULT hr;

        // // 步骤 1: 创建 DirectX 设备和上下文，这是与 GPU 通信的基础
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device_, &feature_level_, &context_);
        if (FAILED(hr)) { return; } // // 初始化失败则直接返回

        // // 步骤 2: 从 D3D11 设备获取 DXGI 设备接口
        ComPtr<IDXGIDevice> dxgi_device;
        hr = device_.As(&dxgi_device);
        if (FAILED(hr)) { return; }

        // // 步骤 3: 获取与设备关联的适配器 (显卡)
        ComPtr<IDXGIAdapter> adapter;
        hr = dxgi_device->GetAdapter(&adapter);
        if (FAILED(hr)) { return; }

        // // 步骤 4: 枚举适配器上的输出 (显示器)，我们只获取第一个 (主显示器)
        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(0, &output);
        if (FAILED(hr)) { return; }

        // // 获取显示器的分辨率
        DXGI_OUTPUT_DESC out_desc;
        output->GetDesc(&out_desc);
        width_ = out_desc.DesktopCoordinates.right - out_desc.DesktopCoordinates.left;
        height_ = out_desc.DesktopCoordinates.bottom - out_desc.DesktopCoordinates.top;

        // // 步骤 5: 获取支持桌面复制API的 IDXGIOutput1 接口
        ComPtr<IDXGIOutput1> output1;
        hr = output.As(&output1);
        if (FAILED(hr)) { return; }

        // // 步骤 6: 创建桌面复制接口，这是实现高效屏幕捕获的核心
        hr = output1->DuplicateOutput(device_.Get(), &duplication_);
        if (FAILED(hr)) { return; }

        // // 如果所有步骤都成功，则将捕获器标记为有效
        valid_ = true;
    }

    // // [核心捕获接口] 捕获一帧屏幕图像
    cv::Mat capture_frame() {
        // // 如果初始化失败或中途出错，则不执行捕获
        if (!valid_) {
            return cv::Mat();
        }

        DXGI_OUTDUPL_FRAME_INFO frame_info;
        ComPtr<IDXGIResource> resource;

        // // 尝试获取下一帧，超时设为 50 毫秒
        HRESULT hr = duplication_->AcquireNextFrame(50, &frame_info, &resource);

        // // 如果超时，说明屏幕内容没有更新，这不是错误，直接返回空 Mat
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            return cv::Mat();
        }
        // // 如果遇到其他错误 (例如全屏应用切换)，则将捕获器设为无效并返回
        if (FAILED(hr)) {
            valid_ = false;
            return cv::Mat();
        }

        // // 将捕获到的通用资源转换为 D3D11 的 2D 纹理
        ComPtr<ID3D11Texture2D> src_tex;
        resource.As(&src_tex);

        // // 获取原始纹理的描述信息 (如尺寸、格式等)
        D3D11_TEXTURE2D_DESC desc;
        src_tex->GetDesc(&desc);

        // // 创建一个 CPU 可读的临时纹理 (Staging Texture)
        // // GPU 上的纹理通常 CPU 无法直接访问，需要一个中介
        D3D11_TEXTURE2D_DESC staging_desc = desc;
        staging_desc.Usage = D3D11_USAGE_STAGING;       // // 用途：作为中介
        staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; // // 权限：CPU 可读
        staging_desc.BindFlags = 0;                      // // 不绑定到渲染管线
        staging_desc.MiscFlags = 0;

        ComPtr<ID3D11Texture2D> staging_tex;
        device_->CreateTexture2D(&staging_desc, nullptr, &staging_tex);

        // // 将 GPU 上的捕获纹理数据复制到 CPU 可读的临时纹理中
        context_->CopyResource(staging_tex.Get(), src_tex.Get());

        // // "映射" 临时纹理到内存，获取指向像素数据的指针
        D3D11_MAPPED_SUBRESOURCE mapped;
        context_->Map(staging_tex.Get(), 0, D3D11_MAP_READ, 0, &mapped);

        // // 创建一个 cv::Mat，并将从 GPU 内存中读取到的像素数据深拷贝进去
        // // 必须进行深拷贝，因为 Unmap 之后 mapped.pData 指针将失效
        cv::Mat frame_bgra(height_, width_, CV_8UC4, mapped.pData, mapped.RowPitch);
        cv::Mat result = frame_bgra.clone();

        // // 清理操作：解除内存映射并释放当前帧，准备下一次捕获
        context_->Unmap(staging_tex.Get(), 0);
        duplication_->ReleaseFrame();

        return result;
    }

    // // [核心捕获接口] 重载：通过绝对坐标(x,y,x1,y1)捕获屏幕的指定区域
    cv::Mat capture_frame(int x, int y, int x1, int y1) {
        // // 首先，捕获整个屏幕
        cv::Mat full_frame = capture_frame();

        // // 如果全屏捕获失败，则返回空Mat
        if (full_frame.empty()) {
            return cv::Mat();
        }

        // // 检查坐标有效性，防止越界
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x1 > full_frame.cols) x1 = full_frame.cols;
        if (y1 > full_frame.rows) y1 = full_frame.rows;

        // // 计算裁剪区域的宽度和高度
        int width = x1 - x;
        int height = y1 - y;

        // // 如果宽度或高度小于等于0，则无法裁剪，返回空Mat
        if (width <= 0 || height <= 0) {
            return cv::Mat();
        }

        // // 定义一个矩形裁剪区域 (Region of Interest)
        cv::Rect roi(x, y, width, height);

        // // 从全屏图像中裁剪出指定区域并返回其拷贝
        return full_frame(roi).clone();
    }


    // // 检查捕获器是否成功初始化
    bool is_valid() const { return valid_; }
    // // 获取捕获的屏幕宽度
    int get_width() const { return width_; }
    // // 获取捕获的屏幕高度
    int get_height() const { return height_; }


private:
    // // DirectX 相关对象
    ComPtr<ID3D11Device> device_;           // // D3D 设备，代表 GPU
    ComPtr<ID3D11DeviceContext> context_;  // // 设备上下文，用于执行渲染命令
    ComPtr<IDXGIOutputDuplication> duplication_; // // 桌面复制接口
    D3D_FEATURE_LEVEL feature_level_;      // // D3D 功能级别

    // // 状态和属性
    bool valid_ = false;   // // 标记捕获器是否有效
    int width_ = 0;        // // 屏幕宽度
    int height_ = 0;       // // 屏幕高度
};

// // 全局 Map，用于存储每个窗口句柄(HWND)与其要显示的 cv::Mat 图像的指针
static std::map<HWND, const cv::Mat*> g_windowImages;
// // [修正] 全局 Map，用于存储窗口名称与其句柄(HWND)的对应关系，避免重复创建
static std::map<std::string, HWND> g_windowHandles;

// // 窗口过程函数：处理所有与窗口相关的消息 (如绘图、关闭等)
LRESULT CALLBACK DisplayWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        // // 当窗口需要重绘时，会收到 WM_PAINT 消息
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps); // // 获取用于绘图的设备上下文句柄 (HDC)

        // // 检查当前窗口是否有对应的图像需要显示
        if (g_windowImages.count(hWnd) && g_windowImages[hWnd] != nullptr && !g_windowImages[hWnd]->empty()) {
            const cv::Mat& img = *g_windowImages[hWnd]; // // 获取图像引用

            // // 准备位图信息头 (BITMAPINFO)，向 Windows GDI 描述图像的格式
            BITMAPINFO bmi = {};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = img.cols;
            bmi.bmiHeader.biHeight = -img.rows; // // 高度为负，表示图像是顶行在上的 (与 cv::Mat 一致)
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = img.channels() * 8; // // 每个像素的位数 (例如 3通道BGR是24位)
            bmi.bmiHeader.biCompression = BI_RGB; // // 不压缩

            // // 获取窗口的当前大小，以便将图像拉伸填充整个窗口
            RECT clientRect;
            GetClientRect(hWnd, &clientRect);

            // // 使用 StretchDIBits 函数将像素数据绘制到窗口上，它会自动处理缩放
            StretchDIBits(hdc, 0, 0, clientRect.right, clientRect.bottom, 0, 0, img.cols, img.rows, img.data, &bmi, DIB_RGB_COLORS, SRCCOPY);
        }
        EndPaint(hWnd, &ps); // // 结束绘图
        return 0;
    }
                 // // 当窗口被销毁时 (例如用户点击关闭按钮)
    case WM_DESTROY: {
        // // 从图像 map 中移除此窗口的条目
        if (g_windowImages.count(hWnd)) {
            g_windowImages.erase(hWnd);
        }

        // // [修正] 遍历句柄 map，找到并移除对应的条目
        for (auto it = g_windowHandles.begin(); it != g_windowHandles.end(); ++it) {
            if (it->second == hWnd) {
                g_windowHandles.erase(it);
                break; // 找到并删除后即可退出循环
            }
        }

        // // 如果这是最后一个窗口，则向主线程发送退出消息
        if (g_windowHandles.empty()) {
            PostQuitMessage(0);
        }
        return 0;
    }
    }
    // // 对于我们不处理的消息，交由系统默认处理
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

// // [核心显示接口] 在指定的 Win32 原生窗口上显示一帧图像 (修正版)
void show_image_in_win32_window(const std::string& windowName, const cv::Mat& frame) {
    // // 检查输入图像是否有效
    if (frame.empty()) return;
    // // 只支持 3 通道 (BGR) 或 4 通道 (BGRA) 图像
    if (frame.channels() != 3 && frame.channels() != 4) return;

    HWND hwnd = nullptr;

    // // 步骤 1: 尝试从我们自己的 map 中查找已存在的窗口句柄
    auto it = g_windowHandles.find(windowName);
    if (it != g_windowHandles.end()) {
        // // 找到了句柄，但要检查窗口是否还存在 (例如，用户可能手动关闭了它)
        if (IsWindow(it->second)) {
            hwnd = it->second;
        }
        else {
            // // 窗口已不存在，从 map 中移除无效句柄
            g_windowHandles.erase(it);
        }
    }

    const char* className = "NativeMatDisplayClass";

    // // 步骤 2: 如果句柄不存在 (hwnd 为空)，则创建新窗口
    if (!hwnd) {
        WNDCLASSEXA wc = { sizeof(WNDCLASSEXA) };
        // // 注册窗口类 (只需注册一次)
        if (!GetClassInfoExA(GetModuleHandle(NULL), className, &wc)) {
            wc.lpfnWndProc = DisplayWndProc; // // 指定处理消息的函数
            wc.hInstance = GetModuleHandle(NULL);
            wc.lpszClassName = className;
            wc.hCursor = LoadCursor(NULL, IDC_ARROW); // // 设置鼠标样式
            wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); // // 设置背景颜色
            if (!RegisterClassExA(&wc)) return; // // 注册失败则返回
        }

        // // 创建窗口实例
        hwnd = CreateWindowExA(0, className, windowName.c_str(), WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT, // // 默认位置
            frame.cols / 2, frame.rows / 2, // // 初始窗口大小设为图像尺寸的一半
            NULL, NULL, GetModuleHandle(NULL), NULL);

        if (!hwnd) return; // // 创建失败则返回

        // // [关键] 将新创建的窗口句柄存入我们的 map 中
        g_windowHandles[windowName] = hwnd;

        // // 显示并更新窗口
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
    }

    // // 步骤 3: 无论窗口是新建的还是已存在的，都更新它要显示的图像指针
    g_windowImages[hwnd] = &frame;
    // // 使窗口的客户区无效，这将强制系统发送一个 WM_PAINT 消息，从而触发重绘
    InvalidateRect(hwnd, NULL, FALSE);

    // // 处理当前线程的消息队列，这是让 GUI 保持响应的关键
    MSG msg = {};
    // // PeekMessage 是非阻塞的，它会处理所有待处理的消息然后立即返回
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            // // 当最后一个窗口关闭时，这里会收到退出消息。
            // // 我们可以通过清空句柄 map 来让主循环知道可以退出了。
            g_windowHandles.clear();
        }
        TranslateMessage(&msg); // // 翻译虚拟键消息
        DispatchMessage(&msg);  // // 将消息分派给窗口过程函数 (DisplayWndProc)
    }
}