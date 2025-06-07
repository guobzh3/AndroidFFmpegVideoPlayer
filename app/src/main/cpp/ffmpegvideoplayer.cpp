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

#define LOG_TAG "MyNativeCode"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Android NDK 相关
JavaVM* javaVM;
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    javaVM = vm;
    return JNI_VERSION_1_6;
}

// 调用的代码
extern "C" JNIEXPORT void JNICALL
Java_com_example_ffmpegvideoplayer_MainActivity_mainDecoder(JNIEnv* env, jobject instance, jstring url) {
    StreamPlayer player(javaVM, url); // 创建一个player 类
    player.start();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ffmpegvideoplayer_MainActivity_convertFloatBufferToArgbPixels(
        JNIEnv* env,
        jobject thiz,
        jobject float_buffer_j,
        jintArray int_array_j,
        jint width,
        jint height,
        jint channels) {

    auto* input_floats = static_cast<float*>(env->GetDirectBufferAddress(float_buffer_j));
    if (input_floats == nullptr) {
        LOGE("convertFloatBufferToArgbPixels: Failed to get direct buffer address for input.");
        return;
    }

    jint* output_ints = env->GetIntArrayElements(int_array_j, nullptr);
    if (output_ints == nullptr) {
        LOGE("convertFloatBufferToArgbPixels: Failed to get int array elements for output.");
        return;
    }

    int num_pixels = width * height;
    jsize output_length = env->GetArrayLength(int_array_j);
    jlong input_capacity = env->GetDirectBufferCapacity(float_buffer_j);

    if (input_capacity < num_pixels * channels * sizeof(float)) {
        LOGE("convertFloatBufferToArgbPixels: Input float buffer too small.");
        env->ReleaseIntArrayElements(int_array_j, output_ints, JNI_ABORT);
        return;
    }

    if (output_length < num_pixels) {
        LOGE("convertFloatBufferToArgbPixels: Output int array too small.");
        env->ReleaseIntArrayElements(int_array_j, output_ints, JNI_ABORT);
        return;
    }

    if (channels != 3) {
        LOGE("convertFloatBufferToArgbPixels: Unsupported channel count %d. Expected 3 (RGB).", channels);
        for (int i = 0; i < num_pixels; ++i) {
            output_ints[i] = 0xFFFF00FF; // Magenta
        }
        env->ReleaseIntArrayElements(int_array_j, output_ints, 0);
        return;
    }

    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0) ? 4 : num_threads;
    std::vector<std::thread> threads;
    int pixels_per_thread = num_pixels / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_pixel = i * pixels_per_thread;
        int end_pixel = (i == num_threads - 1) ? num_pixels : start_pixel + pixels_per_thread;

        threads.emplace_back([=]() {
            for (int p = start_pixel; p < end_pixel; ++p) {
                int r = static_cast<int>(input_floats[p * channels + 0] * 255.0f);
                int g = static_cast<int>(input_floats[p * channels + 1] * 255.0f);
                int b = static_cast<int>(input_floats[p * channels + 2] * 255.0f);

//                r = (r < 0) ? 0 : ((r > 255) ? 255 : r);
//                g = (g < 0) ? 0 : ((g > 255) ? 255 : g);
//                b = (b < 0) ? 0 : ((b > 255) ? 255 : b);

                output_ints[p] = (0xFF << 24) | (r << 16) | (g << 8) | b;
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    env->ReleaseIntArrayElements(int_array_j, output_ints, 0);
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

    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = (num_threads == 0) ? 4 : num_threads; // Fallback to 4 threads if detection fails
    std::vector<std::thread> threads;

    int rows_per_thread = crop_h / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? crop_h : start_row + rows_per_thread;

        threads.emplace_back([=]() {
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
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}


// .\ffmpeg.exe -stream_loop -1 -i G:\Desktop\206.mp4 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
// .\ffmpeg.exe -stream_loop -1 -re -i G:\Desktop\nemo.mp4 -c:v copy -b:v 2000k -framerate 30 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
