#pragma once // ��ֹͷ�ļ����ظ�����

// --- ������ͷ�ļ� ---
#include <windows.h>      // // Win32 API ����ͷ�ļ�
#include <d3d11.h>        // // DirectX 11 ����ͷ�ļ�
#include <dxgi1_2.h>      // // DXGI (DirectX Graphics Infrastructure) ͷ�ļ����������渴��
#include <wrl/client.h>   // // ComPtr ����ָ�룬�����Զ����� COM ������������
#include <opencv2/core.hpp> // // OpenCV ���Ŀ⣬��ʹ�� cv::Mat �ṹ

#include <vector>         // // C++ ��׼�⣬��̬����
#include <string>         // // C++ ��׼�⣬�ַ�������
#include <map>            // // C++ ��׼�⣬���ڽ����ھ��ӳ�䵽ͼ��
#include <iostream>       // // C++ ��׼�⣬���ڿ���̨��������������Ϣ��

// --- ���ӱ�Ҫ�Ŀ� (��������Ŀ�������ֶ����) ---
#pragma comment(lib, "d3d11.lib")    // // ���� DirectX 11 ��
#pragma comment(lib, "dxgi.lib")     // // ���� DXGI ��
#pragma comment(lib, "user32.lib")   // // ���� Windows �û������ (���ڹ���)
#pragma comment(lib, "gdi32.lib")    // // ���� Windows GDI �� (ͼ���豸�ӿڣ����ڻ�ͼ)

// // ʹ�� ComPtr ����ָ��������ռ�
using Microsoft::WRL::ComPtr;


class ScreenCapturer {
public:
    // // ���캯�����ڴ�������ʱ�Զ���ʼ����Ļ������
    ScreenCapturer() {
        HRESULT hr;

        // // ���� 1: ���� DirectX �豸�������ģ������� GPU ͨ�ŵĻ���
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device_, &feature_level_, &context_);
        if (FAILED(hr)) { return; } // // ��ʼ��ʧ����ֱ�ӷ���

        // // ���� 2: �� D3D11 �豸��ȡ DXGI �豸�ӿ�
        ComPtr<IDXGIDevice> dxgi_device;
        hr = device_.As(&dxgi_device);
        if (FAILED(hr)) { return; }

        // // ���� 3: ��ȡ���豸������������ (�Կ�)
        ComPtr<IDXGIAdapter> adapter;
        hr = dxgi_device->GetAdapter(&adapter);
        if (FAILED(hr)) { return; }

        // // ���� 4: ö���������ϵ���� (��ʾ��)������ֻ��ȡ��һ�� (����ʾ��)
        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(0, &output);
        if (FAILED(hr)) { return; }

        // // ��ȡ��ʾ���ķֱ���
        DXGI_OUTPUT_DESC out_desc;
        output->GetDesc(&out_desc);
        width_ = out_desc.DesktopCoordinates.right - out_desc.DesktopCoordinates.left;
        height_ = out_desc.DesktopCoordinates.bottom - out_desc.DesktopCoordinates.top;

        // // ���� 5: ��ȡ֧�����渴��API�� IDXGIOutput1 �ӿ�
        ComPtr<IDXGIOutput1> output1;
        hr = output.As(&output1);
        if (FAILED(hr)) { return; }

        // // ���� 6: �������渴�ƽӿڣ�����ʵ�ָ�Ч��Ļ����ĺ���
        hr = output1->DuplicateOutput(device_.Get(), &duplication_);
        if (FAILED(hr)) { return; }

        // // ������в��趼�ɹ����򽫲��������Ϊ��Ч
        valid_ = true;
    }

    // // [���Ĳ���ӿ�] ����һ֡��Ļͼ��
    cv::Mat capture_frame() {
        // // �����ʼ��ʧ�ܻ���;������ִ�в���
        if (!valid_) {
            return cv::Mat();
        }

        DXGI_OUTDUPL_FRAME_INFO frame_info;
        ComPtr<IDXGIResource> resource;

        // // ���Ի�ȡ��һ֡����ʱ��Ϊ 50 ����
        HRESULT hr = duplication_->AcquireNextFrame(50, &frame_info, &resource);

        // // �����ʱ��˵����Ļ����û�и��£��ⲻ�Ǵ���ֱ�ӷ��ؿ� Mat
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            return cv::Mat();
        }
        // // ��������������� (����ȫ��Ӧ���л�)���򽫲�������Ϊ��Ч������
        if (FAILED(hr)) {
            valid_ = false;
            return cv::Mat();
        }

        // // �����񵽵�ͨ����Դת��Ϊ D3D11 �� 2D ����
        ComPtr<ID3D11Texture2D> src_tex;
        resource.As(&src_tex);

        // // ��ȡԭʼ�����������Ϣ (��ߴ硢��ʽ��)
        D3D11_TEXTURE2D_DESC desc;
        src_tex->GetDesc(&desc);

        // // ����һ�� CPU �ɶ�����ʱ���� (Staging Texture)
        // // GPU �ϵ�����ͨ�� CPU �޷�ֱ�ӷ��ʣ���Ҫһ���н�
        D3D11_TEXTURE2D_DESC staging_desc = desc;
        staging_desc.Usage = D3D11_USAGE_STAGING;       // // ��;����Ϊ�н�
        staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; // // Ȩ�ޣ�CPU �ɶ�
        staging_desc.BindFlags = 0;                      // // ���󶨵���Ⱦ����
        staging_desc.MiscFlags = 0;

        ComPtr<ID3D11Texture2D> staging_tex;
        device_->CreateTexture2D(&staging_desc, nullptr, &staging_tex);

        // // �� GPU �ϵĲ����������ݸ��Ƶ� CPU �ɶ�����ʱ������
        context_->CopyResource(staging_tex.Get(), src_tex.Get());

        // // "ӳ��" ��ʱ�����ڴ棬��ȡָ���������ݵ�ָ��
        D3D11_MAPPED_SUBRESOURCE mapped;
        context_->Map(staging_tex.Get(), 0, D3D11_MAP_READ, 0, &mapped);

        // // ����һ�� cv::Mat�������� GPU �ڴ��ж�ȡ�����������������ȥ
        // // ��������������Ϊ Unmap ֮�� mapped.pData ָ�뽫ʧЧ
        cv::Mat frame_bgra(height_, width_, CV_8UC4, mapped.pData, mapped.RowPitch);
        cv::Mat result = frame_bgra.clone();

        // // �������������ڴ�ӳ�䲢�ͷŵ�ǰ֡��׼����һ�β���
        context_->Unmap(staging_tex.Get(), 0);
        duplication_->ReleaseFrame();

        return result;
    }

    // // [���Ĳ���ӿ�] ���أ�ͨ����������(x,y,x1,y1)������Ļ��ָ������
    cv::Mat capture_frame(int x, int y, int x1, int y1) {
        // // ���ȣ�����������Ļ
        cv::Mat full_frame = capture_frame();

        // // ���ȫ������ʧ�ܣ��򷵻ؿ�Mat
        if (full_frame.empty()) {
            return cv::Mat();
        }

        // // ���������Ч�ԣ���ֹԽ��
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x1 > full_frame.cols) x1 = full_frame.cols;
        if (y1 > full_frame.rows) y1 = full_frame.rows;

        // // ����ü�����Ŀ�Ⱥ͸߶�
        int width = x1 - x;
        int height = y1 - y;

        // // �����Ȼ�߶�С�ڵ���0�����޷��ü������ؿ�Mat
        if (width <= 0 || height <= 0) {
            return cv::Mat();
        }

        // // ����һ�����βü����� (Region of Interest)
        cv::Rect roi(x, y, width, height);

        // // ��ȫ��ͼ���вü���ָ�����򲢷����俽��
        return full_frame(roi).clone();
    }


    // // ��鲶�����Ƿ�ɹ���ʼ��
    bool is_valid() const { return valid_; }
    // // ��ȡ�������Ļ���
    int get_width() const { return width_; }
    // // ��ȡ�������Ļ�߶�
    int get_height() const { return height_; }


private:
    // // DirectX ��ض���
    ComPtr<ID3D11Device> device_;           // // D3D �豸������ GPU
    ComPtr<ID3D11DeviceContext> context_;  // // �豸�����ģ�����ִ����Ⱦ����
    ComPtr<IDXGIOutputDuplication> duplication_; // // ���渴�ƽӿ�
    D3D_FEATURE_LEVEL feature_level_;      // // D3D ���ܼ���

    // // ״̬������
    bool valid_ = false;   // // ��ǲ������Ƿ���Ч
    int width_ = 0;        // // ��Ļ���
    int height_ = 0;       // // ��Ļ�߶�
};

// // ȫ�� Map�����ڴ洢ÿ�����ھ��(HWND)����Ҫ��ʾ�� cv::Mat ͼ���ָ��
static std::map<HWND, const cv::Mat*> g_windowImages;
// // [����] ȫ�� Map�����ڴ洢��������������(HWND)�Ķ�Ӧ��ϵ�������ظ�����
static std::map<std::string, HWND> g_windowHandles;

// // ���ڹ��̺��������������봰����ص���Ϣ (���ͼ���رյ�)
LRESULT CALLBACK DisplayWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        // // ��������Ҫ�ػ�ʱ�����յ� WM_PAINT ��Ϣ
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps); // // ��ȡ���ڻ�ͼ���豸�����ľ�� (HDC)

        // // ��鵱ǰ�����Ƿ��ж�Ӧ��ͼ����Ҫ��ʾ
        if (g_windowImages.count(hWnd) && g_windowImages[hWnd] != nullptr && !g_windowImages[hWnd]->empty()) {
            const cv::Mat& img = *g_windowImages[hWnd]; // // ��ȡͼ������

            // // ׼��λͼ��Ϣͷ (BITMAPINFO)���� Windows GDI ����ͼ��ĸ�ʽ
            BITMAPINFO bmi = {};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = img.cols;
            bmi.bmiHeader.biHeight = -img.rows; // // �߶�Ϊ������ʾͼ���Ƕ������ϵ� (�� cv::Mat һ��)
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = img.channels() * 8; // // ÿ�����ص�λ�� (���� 3ͨ��BGR��24λ)
            bmi.bmiHeader.biCompression = BI_RGB; // // ��ѹ��

            // // ��ȡ���ڵĵ�ǰ��С���Ա㽫ͼ�����������������
            RECT clientRect;
            GetClientRect(hWnd, &clientRect);

            // // ʹ�� StretchDIBits �������������ݻ��Ƶ������ϣ������Զ���������
            StretchDIBits(hdc, 0, 0, clientRect.right, clientRect.bottom, 0, 0, img.cols, img.rows, img.data, &bmi, DIB_RGB_COLORS, SRCCOPY);
        }
        EndPaint(hWnd, &ps); // // ������ͼ
        return 0;
    }
                 // // �����ڱ�����ʱ (�����û�����رհ�ť)
    case WM_DESTROY: {
        // // ��ͼ�� map ���Ƴ��˴��ڵ���Ŀ
        if (g_windowImages.count(hWnd)) {
            g_windowImages.erase(hWnd);
        }

        // // [����] ������� map���ҵ����Ƴ���Ӧ����Ŀ
        for (auto it = g_windowHandles.begin(); it != g_windowHandles.end(); ++it) {
            if (it->second == hWnd) {
                g_windowHandles.erase(it);
                break; // �ҵ���ɾ���󼴿��˳�ѭ��
            }
        }

        // // ����������һ�����ڣ��������̷߳����˳���Ϣ
        if (g_windowHandles.empty()) {
            PostQuitMessage(0);
        }
        return 0;
    }
    }
    // // �������ǲ��������Ϣ������ϵͳĬ�ϴ���
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

// // [������ʾ�ӿ�] ��ָ���� Win32 ԭ����������ʾһ֡ͼ�� (������)
void show_image_in_win32_window(const std::string& windowName, const cv::Mat& frame) {
    // // �������ͼ���Ƿ���Ч
    if (frame.empty()) return;
    // // ֻ֧�� 3 ͨ�� (BGR) �� 4 ͨ�� (BGRA) ͼ��
    if (frame.channels() != 3 && frame.channels() != 4) return;

    HWND hwnd = nullptr;

    // // ���� 1: ���Դ������Լ��� map �в����Ѵ��ڵĴ��ھ��
    auto it = g_windowHandles.find(windowName);
    if (it != g_windowHandles.end()) {
        // // �ҵ��˾������Ҫ��鴰���Ƿ񻹴��� (���磬�û������ֶ��ر�����)
        if (IsWindow(it->second)) {
            hwnd = it->second;
        }
        else {
            // // �����Ѳ����ڣ��� map ���Ƴ���Ч���
            g_windowHandles.erase(it);
        }
    }

    const char* className = "NativeMatDisplayClass";

    // // ���� 2: ������������ (hwnd Ϊ��)���򴴽��´���
    if (!hwnd) {
        WNDCLASSEXA wc = { sizeof(WNDCLASSEXA) };
        // // ע�ᴰ���� (ֻ��ע��һ��)
        if (!GetClassInfoExA(GetModuleHandle(NULL), className, &wc)) {
            wc.lpfnWndProc = DisplayWndProc; // // ָ��������Ϣ�ĺ���
            wc.hInstance = GetModuleHandle(NULL);
            wc.lpszClassName = className;
            wc.hCursor = LoadCursor(NULL, IDC_ARROW); // // ���������ʽ
            wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); // // ���ñ�����ɫ
            if (!RegisterClassExA(&wc)) return; // // ע��ʧ���򷵻�
        }

        // // ��������ʵ��
        hwnd = CreateWindowExA(0, className, windowName.c_str(), WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT, // // Ĭ��λ��
            frame.cols / 2, frame.rows / 2, // // ��ʼ���ڴ�С��Ϊͼ��ߴ��һ��
            NULL, NULL, GetModuleHandle(NULL), NULL);

        if (!hwnd) return; // // ����ʧ���򷵻�

        // // [�ؼ�] ���´����Ĵ��ھ���������ǵ� map ��
        g_windowHandles[windowName] = hwnd;

        // // ��ʾ�����´���
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
    }

    // // ���� 3: ���۴������½��Ļ����Ѵ��ڵģ���������Ҫ��ʾ��ͼ��ָ��
    g_windowImages[hwnd] = &frame;
    // // ʹ���ڵĿͻ�����Ч���⽫ǿ��ϵͳ����һ�� WM_PAINT ��Ϣ���Ӷ������ػ�
    InvalidateRect(hwnd, NULL, FALSE);

    // // ����ǰ�̵߳���Ϣ���У������� GUI ������Ӧ�Ĺؼ�
    MSG msg = {};
    // // PeekMessage �Ƿ������ģ����ᴦ�����д��������ϢȻ����������
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            // // �����һ�����ڹر�ʱ��������յ��˳���Ϣ��
            // // ���ǿ���ͨ����վ�� map ������ѭ��֪�������˳��ˡ�
            g_windowHandles.clear();
        }
        TranslateMessage(&msg); // // �����������Ϣ
        DispatchMessage(&msg);  // // ����Ϣ���ɸ����ڹ��̺��� (DisplayWndProc)
    }
}