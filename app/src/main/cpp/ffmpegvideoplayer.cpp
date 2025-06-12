// Write C++ code here.
//
// Do not forget to dynamically load the C++ library into your application.
//
// For instance,
//
// In MainActivity.java:
//    static {
//       System.loadLibrary("ffmpegvideoplayer");
//    }
//
// Or, in MainActivity.kt:
//    companion object {
//      init {

//         System.loadLibrary("ffmpegvideoplayer")
//      }
//    }
//要在Java中引入C++代码，您需要使用Java Native Interface（JNI）来实现Java和C++之间的交互

extern  "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <jni.h> // java native interface , java-c++接口函数
#include <string>
#include <thread>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <android/log.h>
#include "Streamplayer.h" // streamplayer 类实现在这个头文件中
#include "ThreadPool.h"
#include <future>

#define LOG_TAG "MyNativeCode"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global Thread Pool for JNI functions to avoid thread creation overhead.
// The number of threads can be tuned for optimal performance.
// Using a fixed number (e.g., 2 or 4) can sometimes provide more stable performance
// than relying on hardware_concurrency(), which might fluctuate.
static ThreadPool pool(4);

// Android NDK 相关
JavaVM* javaVM;
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    javaVM = vm;
    return JNI_VERSION_1_6;
}

// 调用的代码
extern "C" JNIEXPORT void JNICALL
Java_com_example_ffmpegvideoplayer_MainActivity_mainDecoder(JNIEnv* env, jobject instance, jstring url, jobject buffer) {
    StreamPlayer player(javaVM, url, buffer); // 创建一个player 类
    player.start();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ffmpegvideoplayer_MainActivity_convertFloatRgbToRgbaUint8(
        JNIEnv* env,
        jobject thiz,
        jobject float_rgb_input_j,
        jobject rgba_uint8_output_j,
        jint width,
        jint height) {

    auto* input_floats = static_cast<float*>(env->GetDirectBufferAddress(float_rgb_input_j));
    if (input_floats == nullptr) {
        LOGE("convertFloatRgbToRgbaUint8: Failed to get direct buffer address for input.");
        return;
    }

    auto* output_bytes = static_cast<uint8_t*>(env->GetDirectBufferAddress(rgba_uint8_output_j));
    if (output_bytes == nullptr) {
        LOGE("convertFloatRgbToRgbaUint8: Failed to get direct buffer address for output.");
        return;
    }

    int num_pixels = width * height;
    jlong input_capacity = env->GetDirectBufferCapacity(float_rgb_input_j);
    jlong output_capacity = env->GetDirectBufferCapacity(rgba_uint8_output_j);

    if (input_capacity < num_pixels * 3 * sizeof(float)) {
        LOGE("convertFloatRgbToRgbaUint8: Input float buffer too small. Has %ld, needs %d", input_capacity, num_pixels * 3 * sizeof(float));
        return;
    }

    if (output_capacity < num_pixels * 4) {
        LOGE("convertFloatRgbToRgbaUint8: Output byte buffer too small. Has %ld, needs %d", output_capacity, num_pixels * 4);
        return;
    }

    const unsigned int num_threads = 2; // Matching the pool size for this task
    std::vector<std::future<void>> futures;
    int pixels_per_thread = num_pixels / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_pixel = i * pixels_per_thread;
        int end_pixel = (i == num_threads - 1) ? num_pixels : start_pixel + pixels_per_thread;

        futures.emplace_back(
            pool.enqueue([=] {
                for (int p = start_pixel; p < end_pixel; ++p) {
                    // Read float RGB
                    float r_float = input_floats[p * 3 + 0];
                    float g_float = input_floats[p * 3 + 1];
                    float b_float = input_floats[p * 3 + 2];

                    // Convert to uint8_t
                    int r_int = static_cast<int>(r_float * 255.0f);
                    int g_int = static_cast<int>(g_float * 255.0f);
                    int b_int = static_cast<int>(b_float * 255.0f);

                    // Directly cast. Note: Clamping might be safer if input isn't guaranteed to be [0,1]
                    uint8_t r = r_int;
                    uint8_t g = g_int;
                    uint8_t b = b_int;

                    // Write uint8_t RGBA
                    output_bytes[p * 4 + 0] = r;
                    output_bytes[p * 4 + 1] = g;
                    output_bytes[p * 4 + 2] = b;
                    output_bytes[p * 4 + 3] = 255; // Alpha
                }
            })
        );
    }

    for (auto& f : futures) {
        f.get(); // Wait for the task to complete
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ffmpegvideoplayer_MainActivity_cropAndNormalizeRgbaToRgbFloat(
        JNIEnv* env,
        jobject thiz,
        jobject input_buffer_j,
        jobject output_buffer_j,
        jint crop_x, jint crop_y,
        jint crop_w, jint crop_h,
        jint input_w) {

    auto* input_buf = static_cast<uint8_t*>(env->GetDirectBufferAddress(input_buffer_j));
    auto* output_buf = static_cast<float*>(env->GetDirectBufferAddress(output_buffer_j));

    if (input_buf == nullptr || output_buf == nullptr) {
        LOGE("cropAndNormalizeRgbaToRgbFloat: Failed to get direct buffer address.");
        return;
    }

    const unsigned int num_threads = 2; // Matching the pool size for this task
    std::vector<std::future<void>> futures;
    int rows_per_thread = crop_h / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? crop_h : start_row + rows_per_thread;

        futures.emplace_back(
            pool.enqueue([=] {
                for (int y = start_row; y < end_row; ++y) {
                    for (int x = 0; x < crop_w; ++x) {
                        int input_pixel_index = ((crop_y + y) * input_w + (crop_x + x)) * 4; // RGBA
                        int output_pixel_index = (y * crop_w + x) * 3; // RGB

                        // RGBA to RGB and normalize
                        output_buf[output_pixel_index + 0] = input_buf[input_pixel_index + 0] / 255.0f; // R
                        output_buf[output_pixel_index + 1] = input_buf[input_pixel_index + 1] / 255.0f; // G
                        output_buf[output_pixel_index + 2] = input_buf[input_pixel_index + 2] / 255.0f; // B
                    }
                }
            })
        );
    }

    for (auto& f : futures) {
        f.get(); // Wait for the task to complete
    }
}


// .\ffmpeg.exe -stream_loop -1 -i G:\Desktop\206.mp4 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
// .\ffmpeg.exe -stream_loop -1 -re -i G:\Desktop\nemo.mp4 -c:v copy -b:v 2000k -framerate 30 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
