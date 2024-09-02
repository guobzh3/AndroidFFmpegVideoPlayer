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

// .\ffmpeg.exe -stream_loop -1 -i G:\Desktop\206.mp4 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
// .\ffmpeg.exe -stream_loop -1 -re -i G:\Desktop\nemo.mp4 -c:v copy -b:v 2000k -framerate 30 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
