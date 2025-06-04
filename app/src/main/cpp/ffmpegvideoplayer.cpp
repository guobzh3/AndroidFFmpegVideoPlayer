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
Java_com_example_ffmpegvideoplayer_MainActivity_convertFloatToArgbPixels(
        JNIEnv* env,
        jobject thiz, /* this */
        jfloatArray floatArray_j,
        jintArray intArray_j,
        jint width,
        jint height,
        jint channels) {

    if (floatArray_j == nullptr || intArray_j == nullptr) {
        LOGE("convertFloatToArgbPixels: Input or output array is null.");
        return;
    }

    jfloat* inputFloats = env->GetFloatArrayElements(floatArray_j, nullptr);
    if (inputFloats == nullptr) {
        LOGE("convertFloatToArgbPixels: Failed to get float array elements.");
        return;
    }

    jint* outputInts = env->GetIntArrayElements(intArray_j, nullptr);
    if (outputInts == nullptr) {
        LOGE("convertFloatToArgbPixels: Failed to get int array elements.");
        env->ReleaseFloatArrayElements(floatArray_j, inputFloats, JNI_ABORT); // Release the acquired float array
        return;
    }

    int numPixels = width * height;
    jsize inputLength = env->GetArrayLength(floatArray_j);
    jsize outputLength = env->GetArrayLength(intArray_j);

    if (inputLength < numPixels * channels) {
        LOGE("convertFloatToArgbPixels: Input float array too small. Expected %d, Got %d", numPixels * channels, inputLength);
        env->ReleaseFloatArrayElements(floatArray_j, inputFloats, JNI_ABORT);
        env->ReleaseIntArrayElements(intArray_j, outputInts, JNI_ABORT);
        return;
    }

    if (outputLength < numPixels) {
        LOGE("convertFloatToArgbPixels: Output int array too small. Expected %d, Got %d", numPixels, outputLength);
        env->ReleaseFloatArrayElements(floatArray_j, inputFloats, JNI_ABORT);
        env->ReleaseIntArrayElements(intArray_j, outputInts, JNI_ABORT);
        return;
    }

    if (channels != 3) {
        LOGE("convertFloatToArgbPixels: Unsupported channel count %d. Expected 3 (RGB).", channels);
        // Fill output with a pattern to indicate error, e.g., magenta
        for (int i = 0; i < numPixels; ++i) {
            outputInts[i] = 0xFFFF00FF; // Magenta
        }
        env->ReleaseFloatArrayElements(floatArray_j, inputFloats, JNI_ABORT);
        env->ReleaseIntArrayElements(intArray_j, outputInts, 0); // Mode 0 to copy back changes if any
        return;
    }

    for (int i = 0; i < numPixels; ++i) {
        // Assuming TFLite output float values are normalized [0,1] or need scaling by 255.
        // The Java code was doing (int)(float_val * 255.0f)
        int r = static_cast<int>(inputFloats[i * channels + 0] * 255.0f);
        int g = static_cast<int>(inputFloats[i * channels + 1] * 255.0f);
        int b = static_cast<int>(inputFloats[i * channels + 2] * 255.0f);

        // Clamp values to [0, 255]
//        r = (r < 0) ? 0 : ((r > 255) ? 255 : r);
//        g = (g < 0) ? 0 : ((g > 255) ? 255 : g);
//        b = (b < 0) ? 0 : ((b > 255) ? 255 : b);

        outputInts[i] = (0xFF << 24) | (r << 16) | (g << 8) | b; // ARGB
    }

    // Release array elements. Mode 0 means copy back changes to intArray_j.
    env->ReleaseFloatArrayElements(floatArray_j, inputFloats, JNI_ABORT); // Input floats are not modified.
    env->ReleaseIntArrayElements(intArray_j, outputInts, 0);
}


// .\ffmpeg.exe -stream_loop -1 -i G:\Desktop\206.mp4 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
// .\ffmpeg.exe -stream_loop -1 -re -i G:\Desktop\nemo.mp4 -c:v copy -b:v 2000k -framerate 30 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
