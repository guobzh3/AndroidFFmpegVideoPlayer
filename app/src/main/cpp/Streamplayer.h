//
// Created by chaibli on 2024/5/21.
//

// JNI 定义了两个关键数据结构，即“JavaVM”和“JNIEnv”。两者本质上都是指向函数表的指针。 JavaVM 提供“调用接口”函数，用于创建和销毁 JavaVM；JNIEnv 提供了大部分 JNI 函数。您的原生函数都会接收 JNIEnv 作为第一个参数
#ifndef FFMPEGVIDEOPLAYER_STREAMPLAYER_H
#define FFMPEGVIDEOPLAYER_STREAMPLAYER_H


extern  "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#include <jni.h>
#include <string>
#include <thread>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <android/log.h>

#define LOG_TAG "MyNativeCode"
#define TIME_TAG "PerfTime" // Consistent with Java tags
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_TIME(...) __android_log_print(ANDROID_LOG_INFO, TIME_TAG, __VA_ARGS__)

class StreamPlayer {
public:
    JNIEnv * env;
    jclass mainActivityClass;
    jmethodID putDataMethod;
    jmethodID updateDecoderTimingsMethod;
    jmethodID updateYuvToRgbTimeMethod;

    AVFormatContext* deFormatc; // 是一个FormatContext
    AVCodecContext* deCodecc; // 是一个CodecContext
    int video_index;
    int frame_decoded_count;
    SwsContext* sws_ctx;

    // multithread yuv2rgb
    int thread_num;
    std::vector<std::thread> process_thread;

    StreamPlayer(JavaVM* javaVM, jstring url) {
        // javaVM ： java线程的句柄
        // 要在jni代码的线程中调用java代码的方法，必须把当前线程连接到VM中，获取到一个[JNIEnv*].
        // 该 JNIEnv 将用于线程本地存储。因此，您无法在线程之间共享 JNIEnv。如果代码段无法通过其他方法获取其 JNIEnv，您应该共享 JavaVM，并使用 GetEnv 发现线程的 JNIEnv。（假设该线程包含一个 JNIEnv；请参阅下面的 AttachCurrentThread。
        // 将jvm附加到当前线程，后面才可以进行JNI调用
        // 通过 JNI 附加的线程必须在退出之前调用 DetachCurrentThread()
//        LOGI("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        if (javaVM->AttachCurrentThread(&env, nullptr) != 0) {
            LOGI("Failed to attach current thread");
            throw std::runtime_error("Failed to attach current thread");
        }
        // 找到这个java中中class ? 这个有什么用？
        mainActivityClass = env->FindClass("com/example/ffmpegvideoplayer/MainActivity");
        if (mainActivityClass == nullptr) {
            // ... (error handling as before)
            throw std::runtime_error("Failed to find class MainActivity");
        }

        // Make it a global reference to be safe across threads, though in this specific
        // implementation it might not be strictly necessary as JNIEnv is thread-local.
        mainActivityClass = (jclass)env->NewGlobalRef(mainActivityClass);


        // Get method IDs for all callbacks
        putDataMethod = env->GetStaticMethodID(mainActivityClass, "putData", "([B)V");
        updateDecoderTimingsMethod = env->GetStaticMethodID(mainActivityClass, "updateDecoderTimings", "(JJ)V");
        updateYuvToRgbTimeMethod = env->GetStaticMethodID(mainActivityClass, "updateYuvToRgbTime", "(J)V");

        if (putDataMethod == nullptr || updateDecoderTimingsMethod == nullptr || updateYuvToRgbTimeMethod == nullptr) {
            LOGI("Failed to find one or more callback methods in MainActivity");
            if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
            }
            javaVM->DetachCurrentThread();
            throw std::runtime_error("Failed to find JNI callback methods");
        }

        this->frame_decoded_count = 0;
        this->sws_ctx = nullptr;
        this->deFormatc = createFormatc(url); // 用于读取packet av_read_frame(this->deFormatc, input_packet);
        this->deCodecc = createCodecc(this->deFormatc); // 用于对packet进行解码，avcodec_send_packet(this->deCodecc, received_packet); avcodec_receive_frame(this->deCodecc, input_frame);

        // multithread yuv2rgb
        this->thread_num = 2;
        this->process_thread = std::vector<std::thread>(this->thread_num);
    }

    AVFormatContext* createFormatc(jstring url) {
        // 是一个网址链接
        const char* video_address = env->GetStringUTFChars(url, nullptr);
        LOGI("%s", video_address);
        // create decoder
        AVFormatContext* av_formatc = avformat_alloc_context(); // avformat_alloc_context();
        if (!av_formatc) {
            LOGI("Failed to alloc memory for avformat");
            throw std::runtime_error("Failed to alloc memory for avformat");
        }
        // setting params 配置参数
        AVDictionary* opts = nullptr; // 是ffmpeg中用来存储选项的结构体
        av_dict_set(&opts, "rtsp_transport", "tcp", 0); // 指定tcp作为RTSP的传输协议
        // open video 打开文件（注意设置了option参数）
//        int ret = avformat_open_input(av_formatc,video_address, nullptr , &opts);
        int ret = avformat_open_input(&av_formatc, video_address, nullptr, &opts);
        if (ret != 0) {
            LOGI("Failed to open input file");
            throw std::runtime_error("Failed to open input file");
        }
        // find the input stream
        // it will be blocked if broadcaster doesn't send the stream
        LOGI("Waiting for the stream ...");
        // 获取视频流信息，保存在 av_formatc 中
        ret = avformat_find_stream_info(av_formatc, nullptr);
        if (ret != 0) {
            LOGI("Failed to get stream info");
            throw std::runtime_error("Failed to get stream info");
        }
        return av_formatc; // 返回了avformatc
    }

    AVCodecContext* createCodecc(AVFormatContext* avFormatc) {
        if (avFormatc == nullptr) {
            LOGI("de_formatc is nullptr!");
            throw std::runtime_error("de_formatc is nullptr!");
        }
        AVStream* de_stream = nullptr;
        // find stream index（获取视频流的stream）
        for (int i = 0; i < avFormatc->nb_streams; i++) {
            if (avFormatc->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_index = i;
                de_stream = avFormatc->streams[i];
                break;
            }
        }
        // find de_codec by codec_id
        const AVCodec* de_codec = avcodec_find_decoder(de_stream->codecpar->codec_id);
        if (!de_codec) {
            LOGI("Failed to find the de_codec");
            throw std::runtime_error("Failed to find the de_codec");
        }
        // use the de_codec to create de_codec_context
        AVCodecContext* de_codecc = avcodec_alloc_context3(de_codec); //创建一个适配该codec的 codec_context
        if (!de_codecc) {
            LOGI("Failed to alloc memory for de_codec context");
            throw std::runtime_error("Failed to alloc memory for de_codec context");
        }
        // copy the params from de_stream to de_codec_context
        //
        int ret = avcodec_parameters_to_context(de_codecc, de_stream->codecpar);
        if (ret < 0) {
            LOGI("Failed to copy the params to de_codec context");
            throw std::runtime_error("Failed to copy the params to de_codec context");
        }
        de_codecc->thread_count = 16;
        ret = avcodec_open2(de_codecc, de_codec, nullptr); // 打开编码器
        if (ret < 0) {
            LOGI("Failed to open de_codecc");
            throw std::runtime_error("Failed to open de_codecc");
        }
        LOGI("Successfully initial AVCodecContext");
        return de_codecc;
    }

    void start() {
        AVPacket* input_packet = av_packet_alloc();
        int packet_num = 0;
        bool stop = false;
        while (!stop) {
            auto loopStart = std::chrono::high_resolution_clock::now();

            auto readStart = std::chrono::high_resolution_clock::now();
            int ret = av_read_frame(this->deFormatc, input_packet);
            auto readEnd = std::chrono::high_resolution_clock::now();
            auto readDuration = std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart).count();

            if (ret < 0) {
                LOGI("av_read_frame returned %d, stopping.", ret);
                stop = true;
            } else {
                if (input_packet->stream_index == this->video_index) {
                    ret = decoding(input_packet);
                    if (ret < 0) {
                        LOGI("Decoding Error");
                    }
                    packet_num++;
                }
                // Always unref the packet
                auto loopEnd = std::chrono::high_resolution_clock::now();
                auto loopDuration = std::chrono::duration_cast<std::chrono::microseconds>(loopEnd - loopStart).count();
                
                // Call JNI method to update timings
                env->CallStaticVoidMethod(mainActivityClass, updateDecoderTimingsMethod, (jlong)loopDuration, (jlong)readDuration);
            }

            auto loopEnd = std::chrono::high_resolution_clock::now();
            auto loopDuration = std::chrono::duration_cast<std::chrono::microseconds>(loopEnd - loopStart).count();
            // LOG_TIME("Decoder - Total: %.2f ms | Read: %.2f ms", loopDuration / 1000.0, readDuration / 1000.0);
        }
        // flush decoder
        int ret = decoding(nullptr);
        if (ret < 0) {
            LOGI("Flush decoder Error");
        }
        av_packet_free(&input_packet);
        LOGI("Receiving done!");
    }
    /**
     * 对packet进行解码
     * @param received_packet
     * @return
     */
    int decoding(AVPacket* received_packet) {
        AVFrame* input_frame = av_frame_alloc();
        int ret = avcodec_send_packet(this->deCodecc, received_packet);
        if (ret < 0) {
            LOGI("Error while sending packet to decoder");
            return -1;
        }
        while (ret >= 0) {
            ret = avcodec_receive_frame(this->deCodecc, input_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            else if (ret < 0) {
                LOGI("Error while receiving frame from decoder");
                return ret;
            }

            // 将解码得到的avframe 从 YUV 转化为 ARGB8888格式
            ret = avFrameYUV420ToARGB8888(input_frame);



            if (ret < 0) {
                LOGI("Error! avFrameYUV420ToARGB8888");
            }
            frame_decoded_count++;
            
            if (frame_decoded_count % 30 == 0) LOGI("frame decoded count %d, wdith = %d, height = %d", frame_decoded_count, input_frame->width, input_frame->height);

            av_frame_unref(input_frame);
        }
        av_frame_free(&input_frame);

        return 0;
    }

    // AVFrame 的内存布局
    /*             <------------Y-linesize----------->
      *             <-------------width------------>
      *             -----------------------------------
      *             |                              |  |
      *             |                              |  |
      *   height    |              Y               |  |
      *             |                data[0]       |  |
      *             |                              |  |
      *             |                              |  |
      *             -----------------------------------
      *             |             |  |             |  |
      * height / 2  |      U      |  |      V      |  |
      *             |    data[1]  |  | data[2]     |  |
      *             -----------------------------------
      *             <---U-linesize--> <--V-linesize--->
      *             <---U-width--->   <--V-width--->
  */
    int avFrameYUV420ToARGB8888(AVFrame* frame) {
        int width = frame->width;
        int height = frame->height;

        if (this->sws_ctx == nullptr) {
            this->sws_ctx = sws_getContext(width, height, (AVPixelFormat)frame->format,
                                     width, height, AV_PIX_FMT_RGBA,
                                     SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!this->sws_ctx) {
                LOGI("Failed to create sws context");
                return -1;
            }
        }

        // 4 bytes per pixel for RGBA
        jbyteArray outFrame = env->NewByteArray(width * height * 4);
        if (outFrame == nullptr) {
            LOGI("Failed to allocate memory for outFrame jbyteArray");
            return -1;
        }

        jbyte* outData = env->GetByteArrayElements(outFrame, nullptr);
        if (outData == nullptr) {
            LOGI("outData is nullptr, failed to get byte array elements");
            env->DeleteLocalRef(outFrame);
            return -1;
        }

        uint8_t* dst_data[1] = { (uint8_t*)outData };
        int dst_linesize[1] = { width * 4 };

        auto startTime = std::chrono::high_resolution_clock::now();

        sws_scale(this->sws_ctx, (const uint8_t* const*)frame->data, frame->linesize, 0, height,
                  dst_data, dst_linesize);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto durationUs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        
        // Call JNI method to update YUV->RGB time
        env->CallStaticVoidMethod(mainActivityClass, updateYuvToRgbTimeMethod, (jlong)durationUs);

        env->ReleaseByteArrayElements(outFrame, outData, 0); // Mode 0: copy back and free the buffer
        env->CallStaticVoidMethod(mainActivityClass, putDataMethod, outFrame);
        env->DeleteLocalRef(outFrame);

        return 0;
    }

    static int YUV2RGB(int y, int u , int v) {
        int kMaxChannelValue = 262143;
        // Adjust and check YUV values
        y = (y - 16) < 0 ? 0 : (y - 16);
        u -= 128;
        v -= 128;
        // This is the floating point equivalent. We do the conversion in integer
        // because some Android devices do not have floating point in hardware.
        // nR = (int)(1.164 * nY + 1.596 * nV);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 2.018 * nU);
        // 这里取的系数为1024，两边都同时乘上1024
        int y1192 = 1192 * y;
        int r = (y1192 + 1634 * v);
        int g = (y1192 - 833 * v - 400 * u);
        int b = (y1192 + 2066 * u);
        // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
        // KMaxChannelValue = 262143 即 2^18，原始范围应该限制在[0,255]之间，由于换成整数乘了1024故在[0,2^18]
        r = r > kMaxChannelValue ? kMaxChannelValue : (r < 0 ? 0 : r);
        g = g > kMaxChannelValue ? kMaxChannelValue : (g < 0 ? 0 : g);
        b = b > kMaxChannelValue ? kMaxChannelValue : (b < 0 ? 0 : b);
        // 本来应该是 int rgb = (0xFF << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF); ARGB
        // 但是由于转换时乘了1024，需要除以1024，所以是如下的表达式
        return 0xff000000 | ((r << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((b >> 10) & 0xff);
    }
};


#endif //FFMPEGVIDEOPLAYER_STREAMPLAYER_H