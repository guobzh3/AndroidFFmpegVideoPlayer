// streamplayer.h
//
// Created by chaibli on 2024/5/21.
//

#ifndef FFMPEGVIDEOPLAYER_STREAMPLAYER_H
#define FFMPEGVIDEOPLAYER_STREAMPLAYER_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <pthread.h> // For naming threads
}

#include <jni.h>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <android/log.h>

#define LOG_TAG "MyNativeCode"
#define TIME_TAG "PerfTime"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_TIME(...) __android_log_print(ANDROID_LOG_INFO, TIME_TAG, __VA_ARGS__)

// A thread-safe queue for AVPacket pointers, serving as a jitter buffer.
const int PACKET_QUEUE_MAX_SIZE = 150; // Buffer for ~5 seconds of video at 30fps

class PacketQueue {
public:
    PacketQueue() = default;

    bool push(AVPacket* packet) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_not_full_.wait(lock, [this] {
            return abort_request_ || queue_.size() < PACKET_QUEUE_MAX_SIZE;
        });

        if (abort_request_) {
            return false;
        }

        queue_.push(packet);
        cond_not_empty_.notify_one();
        return true;
    }

    bool pop(AVPacket** packet) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_not_empty_.wait(lock, [this] {
            return abort_request_ || !queue_.empty();
        });

        if (abort_request_ && queue_.empty()) {
            return false;
        }

        *packet = queue_.front();
        queue_.pop();
        cond_not_full_.notify_one();
        return true;
    }

    void abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        abort_request_ = true;
        cond_not_empty_.notify_all();
        cond_not_full_.notify_all();
    }

private:
    std::queue<AVPacket*> queue_;
    std::mutex mutex_;
    std::condition_variable cond_not_empty_;
    std::condition_variable cond_not_full_;
    std::atomic<bool> abort_request_{false};
};

class StreamPlayer {
public:
    // JNI 定义了两个关键数据结构，即“JavaVM”和“JNIEnv”。两者本质上都是指向函数表的指针。
    // JavaVM 提供“调用接口”函数，用于创建和销毁 JavaVM。
    // *** FIX: Constructor now accepts JNIEnv* directly and does not Attach/Detach ***
    StreamPlayer(JavaVM* javaVM, JNIEnv* env, jstring url, jobject buffer) : jvm_(javaVM) {
        try {
            mainActivityClass = env->FindClass("com/example/ffmpegvideoplayer/MainActivity");
            if (mainActivityClass == nullptr) {
                throw std::runtime_error("Failed to find class MainActivity");
            }
            // Make it a global reference to be safe across threads.
            mainActivityClass = (jclass)env->NewGlobalRef(mainActivityClass);

            onFrameReadyMethod = env->GetStaticMethodID(mainActivityClass, "onFrameReady", "()V");
            updateDecoderTimingsMethod = env->GetStaticMethodID(mainActivityClass, "updateDecoderTimings", "(JJ)V");
            updateYuvToRgbTimeMethod = env->GetStaticMethodID(mainActivityClass, "updateYuvToRgbTime", "(J)V");

            if (!onFrameReadyMethod || !updateDecoderTimingsMethod || !updateYuvToRgbTimeMethod) {
                throw std::runtime_error("Failed to find one or more JNI callback methods");
            }

            output_buffer_ = (uint8_t*)env->GetDirectBufferAddress(buffer);
            deFormatc_ = createFormatc(env, url);
            deCodecc_ = createCodecc(deFormatc_);

            reusable_frame_ = av_frame_alloc();
            if (!reusable_frame_) {
                throw std::runtime_error("Failed to allocate reusable AVFrame");
            }
        } catch (const std::runtime_error& e) {
            // Robust cleanup in case of constructor failure
            LOGI("Exception during StreamPlayer construction: %s", e.what());
            cleanup(env);
            throw; // Re-throw to notify Java side of the failure
        }
        LOGI("StreamPlayer initialized successfully.");
    }

    ~StreamPlayer() {
        stop(); // Ensure everything is stopped and cleaned up.
        JNIEnv* env = nullptr;
        if (jvm_->AttachCurrentThread(&env, nullptr) == 0) {
            cleanup(env);
            jvm_->DetachCurrentThread();
        }
    }

    void start() {
        if (!player_stopped_.exchange(false)) {
            // Already running
            return;
        }
        LOGI("Starting player threads...");
        read_thread_ = std::thread(&StreamPlayer::read_loop, this);
        decode_thread_ = std::thread(&StreamPlayer::decode_loop, this);
    }

    void stop() {
        if (player_stopped_.exchange(true)) {
            return; // Already stopped
        }
        LOGI("Stopping StreamPlayer...");

        packet_queue_.abort();

        if (read_thread_.joinable()) {
            read_thread_.join();
        }
        if (decode_thread_.joinable()) {
            decode_thread_.join();
        }
        LOGI("Player threads stopped.");
    }

private:
    // JNI and threading members
    JavaVM* jvm_;
    jclass mainActivityClass = nullptr;
    jmethodID onFrameReadyMethod = nullptr;
    jmethodID updateDecoderTimingsMethod = nullptr;
    jmethodID updateYuvToRgbTimeMethod = nullptr;
    std::thread read_thread_;
    std::thread decode_thread_;
    std::atomic<bool> player_stopped_{true};

    // FFmpeg members
    AVFormatContext* deFormatc_ = nullptr;
    AVCodecContext* deCodecc_ = nullptr;
    AVFrame* reusable_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int video_index_ = -1;
    PacketQueue packet_queue_;

    // Buffer for output
    uint8_t* output_buffer_ = nullptr;
    int frame_decoded_count_ = 0;

    void cleanup(JNIEnv* env) {
        if (reusable_frame_) av_frame_free(&reusable_frame_);
        if (sws_ctx_) sws_freeContext(sws_ctx_);
        if (deCodecc_) avcodec_free_context(&deCodecc_);
        if (deFormatc_) avformat_close_input(&deFormatc_);
        if (mainActivityClass) env->DeleteGlobalRef(mainActivityClass);

        reusable_frame_ = nullptr;
        sws_ctx_ = nullptr;
        deCodecc_ = nullptr;
        deFormatc_ = nullptr;
        mainActivityClass = nullptr;
    }

    // Producer thread: reads packets from network and puts them into the queue
    void read_loop() {
        pthread_setname_np(pthread_self(), "NativeReadThread");
        LOGI("Read thread started.");
        while (!player_stopped_) {
            AVPacket* packet = av_packet_alloc();
            if (!packet) {
                LOGI("Failed to allocate AVPacket in read_loop.");
                break;
            }

            int ret = av_read_frame(deFormatc_, packet);
            if (ret < 0) {
                LOGI("av_read_frame returned %d, end of stream or error. Aborting.", ret);
                av_packet_free(&packet);
                packet_queue_.abort(); // Signal consumer to stop
                break;
            }

            if (packet->stream_index == video_index_) {
                if (!packet_queue_.push(packet)) {
                    av_packet_free(&packet);
                    break;
                }
            } else {
                av_packet_free(&packet);
            }
        }
        LOGI("Read thread finished.");
    }

    // Consumer thread: takes packets from queue, decodes, and renders
    void decode_loop() {
        pthread_setname_np(pthread_self(), "NativeDecodeThread");
        JNIEnv* env;
        if (jvm_->AttachCurrentThread(&env, nullptr) != 0) {
            LOGI("Failed to attach decode_thread to JVM");
            return;
        }
        LOGI("Decode thread started.");

        while (!player_stopped_) {
            AVPacket* packet = nullptr;

            if (!packet_queue_.pop(&packet)) {
                break;
            }

            decoding(env, packet);
            av_packet_free(&packet);
        }

        decoding(env, nullptr); // Flush decoder

        jvm_->DetachCurrentThread();
        LOGI("Decode thread finished.");
    }

    AVFormatContext* createFormatc(JNIEnv* env, jstring url) {
        const char* video_address = env->GetStringUTFChars(url, nullptr);
        LOGI("Opening URL: %s", video_address);
        AVFormatContext* av_formatc = avformat_alloc_context();
        if (!av_formatc) {
            env->ReleaseStringUTFChars(url, video_address);
            throw std::runtime_error("Failed to alloc memory for avformat");
        }

        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
        av_dict_set(&opts, "buffer_size", "2048000", 0);
        av_dict_set(&opts, "max_delay", "500000", 0);
        av_dict_set(&opts, "stimeout", "5000000", 0);

        int ret = avformat_open_input(&av_formatc, video_address, nullptr, &opts);
        env->ReleaseStringUTFChars(url, video_address);
        av_dict_free(&opts);

        if (ret != 0) {
            avformat_free_context(av_formatc);
            throw std::runtime_error("Failed to open input file");
        }

        LOGI("Waiting for the stream info...");
        if (avformat_find_stream_info(av_formatc, nullptr) < 0) {
            avformat_close_input(&av_formatc);
            throw std::runtime_error("Failed to get stream info");
        }
        return av_formatc;
    }

    AVCodecContext* createCodecc(AVFormatContext* avFormatc) {
        AVStream* de_stream = nullptr;
        int v_idx = av_find_best_stream(avFormatc, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (v_idx < 0) {
            throw std::runtime_error("Failed to find video stream");
        }
        video_index_ = v_idx;
        de_stream = avFormatc->streams[video_index_];
        LOGI("Stream codec ID: %s", avcodec_get_name(de_stream->codecpar->codec_id));
        const AVCodec* de_codec = nullptr;
        if (de_stream->codecpar->codec_id == AV_CODEC_ID_H264) {
            LOGI("Stream is H.264, trying to find h264_mediacodec decoder...");
            de_codec = avcodec_find_decoder_by_name("h264_mediacodec");
            if (de_codec) LOGI("Found hardware decoder: h264_mediacodec");
        }


        if (de_stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
            LOGI("Stream is H.265, trying to find h265_mediacodec decoder...");
            de_codec = avcodec_find_decoder_by_name("hevc_mediacodec");
            if (de_codec) LOGI("Found hardware decoder: h265_mediacodec");
        }

        if (!de_codec) {
            LOGI("Falling back to default software decoder.");
            de_codec = avcodec_find_decoder(de_stream->codecpar->codec_id);
        }
        if (!de_codec) {
            throw std::runtime_error("Failed to find any suitable decoder");
        }

        AVCodecContext* de_codecc = avcodec_alloc_context3(de_codec);
        if (!de_codecc) {
            throw std::runtime_error("Failed to alloc memory for de_codec context");
        }
        if (avcodec_parameters_to_context(de_codecc, de_stream->codecpar) < 0) {
            avcodec_free_context(&de_codecc);
            throw std::runtime_error("Failed to copy params to de_codec context");
        }

        de_codecc->thread_count = 4;
        if (avcodec_open2(de_codecc, de_codec, nullptr) < 0) {
            avcodec_free_context(&de_codecc);
            throw std::runtime_error("Failed to open de_codecc");
        }
        LOGI("Successfully initialized AVCodecContext with decoder: %s", de_codec->name);
        return de_codecc;
    }

    int decoding(JNIEnv* env, AVPacket* received_packet) {
        int ret = avcodec_send_packet(deCodecc_, received_packet);
        if (ret < 0) {
            return -1;
        }
        while (ret >= 0) {
            ret = avcodec_receive_frame(deCodecc_, reusable_frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                return ret;
            }

            const int64_t FRAME_MIN_DURATION_US = 32000;
            auto frame_start_time = std::chrono::high_resolution_clock::now();

            avFrameYUV420ToARGB8888(env, reusable_frame_);
            frame_decoded_count_++;

            auto frame_end_time = std::chrono::high_resolution_clock::now();
            auto processing_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(frame_end_time - frame_start_time).count();

            if (processing_duration_us < FRAME_MIN_DURATION_US) {
                auto sleep_duration = std::chrono::microseconds(FRAME_MIN_DURATION_US - processing_duration_us);
                std::this_thread::sleep_for(sleep_duration);
            }

            av_frame_unref(reusable_frame_);
        }
        return 0;
    }

    int avFrameYUV420ToARGB8888(JNIEnv* env, AVFrame* frame) {
        int width = frame->width;
        int height = frame->height;

        if (sws_ctx_ == nullptr) {
            sws_ctx_ = sws_getContext(width, height, (AVPixelFormat)frame->format,
                                      width, height, AV_PIX_FMT_RGBA,
                                      SWS_BICUBIC, nullptr, nullptr, nullptr);
            if (!sws_ctx_) return -1;
        }

        uint8_t* dst_data[1] = { output_buffer_ };
        int dst_linesize[1] = { width * 4 };

        auto startTime = std::chrono::high_resolution_clock::now();
        sws_scale(sws_ctx_, (const uint8_t* const*)frame->data, frame->linesize, 0, height,
                  dst_data, dst_linesize);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto durationUs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        env->CallStaticVoidMethod(mainActivityClass, updateYuvToRgbTimeMethod, (jlong)durationUs);
        env->CallStaticVoidMethod(mainActivityClass, onFrameReadyMethod);

        return 0;
    }
};

#endif //FFMPEGVIDEOPLAYER_STREAMPLAYER_H