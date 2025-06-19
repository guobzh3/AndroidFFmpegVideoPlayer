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
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <jni.h>
#include <string>
#include <thread>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <android/log.h>
#include "Streamplayer.h"
#include "ThreadPool.h"
#include <future>
#include <arm_neon.h>
#include <algorithm> // For std::min/max

#define LOG_TAG "MyNativeCode"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global Thread Pool: Number of threads is set to half the available cores.
static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency() / 2));

static JavaVM* g_jvm = nullptr;
static StreamPlayer* player = nullptr;

// Called when your library is loaded
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

// --- Unchanged Player Control Functions ---

extern "C" JNIEXPORT void JNICALL
Java_com_example_ffmpegvideoplayer_MainActivity_nativeInitPlayer(
        JNIEnv* env,
        jobject /* this */,
        jstring url,
        jobject buffer) {
    av_log_set_level(AV_LOG_INFO);
    if (player) {
        delete player;
        player = nullptr;
    }
    try {
        player = new StreamPlayer(g_jvm, env, url, buffer);
    } catch (const std::runtime_error& e) {
        LOGI("Failed to create StreamPlayer: %s", e.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ffmpegvideoplayer_MainActivity_nativeStartPlayer(
        JNIEnv* env,
        jobject /* this */) {
    if (player) {
        player->start();
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ffmpegvideoplayer_MainActivity_nativeStopPlayer(
        JNIEnv* env,
        jobject /* this */) {
    if (player) {
        delete player;
        player = nullptr;
    }
}

// --- Optimized Image Processing Functions ---

/**
 * @brief Draws a white border on an RGBA image buffer.
 * @param output_bytes Pointer to the RGBA image data.
 * @param width Image width.
 * @param height Image height.
 * @param thickness Border thickness in pixels.
 */
void draw_white_border(uint8_t* output_bytes, int width, int height, int thickness) {
    if (thickness <= 0) return;

    int border_w = std::min(thickness, width / 2);
    int border_h = std::min(thickness, height / 2);
    uint32_t white_pixel = 0xFFFFFFFF; // RGBA(255, 255, 255, 255)

    // Top and Bottom borders
    for (int y = 0; y < border_h; ++y) {
        for (int x = 0; x < width; ++x) {
            reinterpret_cast<uint32_t*>(output_bytes)[y * width + x] = white_pixel;
            reinterpret_cast<uint32_t*>(output_bytes)[(height - 1 - y) * width + x] = white_pixel;
        }
    }

    // Left and Right borders (avoiding corners)
    for (int y = border_h; y < height - border_h; ++y) {
        for (int x = 0; x < border_w; ++x) {
            reinterpret_cast<uint32_t*>(output_bytes)[y * width + x] = white_pixel;
            reinterpret_cast<uint32_t*>(output_bytes)[y * width + (width - 1 - x)] = white_pixel;
        }
    }
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
    if (!input_floats) {
        LOGE("convertFloatRgbToRgbaUint8: Input buffer is null.");
        return;
    }

    auto* output_bytes = static_cast<uint8_t*>(env->GetDirectBufferAddress(rgba_uint8_output_j));
    if (!output_bytes) {
        LOGE("convertFloatRgbToRgbaUint8: Output buffer is null.");
        return;
    }

    int num_pixels = width * height;
    // Buffer capacity checks (unchanged, but essential)
    if (env->GetDirectBufferCapacity(float_rgb_input_j) < num_pixels * 3 * sizeof(float) ||
        env->GetDirectBufferCapacity(rgba_uint8_output_j) < num_pixels * 4) {
        LOGE("convertFloatRgbToRgbaUint8: Buffer capacity check failed.");
        return;
    }

    const unsigned int num_threads = 1;
    std::vector<std::future<void>> futures;
    int pixels_per_thread = num_pixels / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_pixel = i * pixels_per_thread;
        int end_pixel = (i == num_threads - 1) ? num_pixels : start_pixel + pixels_per_thread;

        futures.emplace_back(
                pool.enqueue([=] {
                    const float32x4_t v255 = vdupq_n_f32(255.0f);
                    const uint8x8_t alpha_channel = vdup_n_u8(255);

                    int p = start_pixel;
                    // Process 8 pixels per iteration using NEON
                    for (; p <= end_pixel - 8; p += 8) {
                        // 1. Load 8 sets of RGB floats (de-interleaving)
                        float32x4x3_t rgb_low  = vld3q_f32(input_floats + (p + 0) * 3);
                        float32x4x3_t rgb_high = vld3q_f32(input_floats + (p + 4) * 3);

                        // 2. Multiply by 255
                        float32x4_t r_f32_low = vmulq_f32(rgb_low.val[0], v255);
                        float32x4_t g_f32_low = vmulq_f32(rgb_low.val[1], v255);
                        float32x4_t b_f32_low = vmulq_f32(rgb_low.val[2], v255);
                        float32x4_t r_f32_high = vmulq_f32(rgb_high.val[0], v255);
                        float32x4_t g_f32_high = vmulq_f32(rgb_high.val[1], v255);
                        float32x4_t b_f32_high = vmulq_f32(rgb_high.val[2], v255);

                        // 3. Convert float to uint32
                        uint32x4_t r_u32_low = vcvtq_u32_f32(r_f32_low);
                        uint32x4_t g_u32_low = vcvtq_u32_f32(g_f32_low);
                        uint32x4_t b_u32_low = vcvtq_u32_f32(b_f32_low);
                        uint32x4_t r_u32_high = vcvtq_u32_f32(r_f32_high);
                        uint32x4_t g_u32_high = vcvtq_u32_f32(g_f32_high);
                        uint32x4_t b_u32_high = vcvtq_u32_f32(b_f32_high);

                        // 4. Narrow uint32 to uint16, then to uint8 with saturation
                        uint8x8_t r_u8 = vqmovn_u16(vcombine_u16(vqmovn_u32(r_u32_low), vqmovn_u32(r_u32_high)));
                        uint8x8_t g_u8 = vqmovn_u16(vcombine_u16(vqmovn_u32(g_u32_low), vqmovn_u32(g_u32_high)));
                        uint8x8_t b_u8 = vqmovn_u16(vcombine_u16(vqmovn_u32(b_u32_low), vqmovn_u32(b_u32_high)));

                        // 5. Store as interleaved RGBA
                        uint8x8x4_t rgba_out;
                        rgba_out.val[0] = r_u8;
                        rgba_out.val[1] = g_u8;
                        rgba_out.val[2] = b_u8;
                        rgba_out.val[3] = alpha_channel;
                        vst4_u8(output_bytes + p * 4, rgba_out);
                    }

                    // Process remaining pixels
                    for (; p < end_pixel; ++p) {
                        output_bytes[p * 4 + 0] = std::min(255.0f, std::max(0.0f, input_floats[p * 3 + 0] * 255.0f));
                        output_bytes[p * 4 + 1] = std::min(255.0f, std::max(0.0f, input_floats[p * 3 + 1] * 255.0f));
                        output_bytes[p * 4 + 2] = std::min(255.0f, std::max(0.0f, input_floats[p * 3 + 2] * 255.0f));
                        output_bytes[p * 4 + 3] = 255;
                    }
                })
        );
    }

    for (auto& f : futures) {
        f.get();
    }

    // Draw a 2px white border after conversion is complete
    draw_white_border(output_bytes, width, height, 2);
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

    if (!input_buf || !output_buf) {
        LOGE("cropAndNormalizeRgbaToRgbFloat: Buffer address is null.");
        return;
    }

    const unsigned int num_threads = 1;
    std::vector<std::future<void>> futures;
    int rows_per_thread = crop_h / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? crop_h : start_row + rows_per_thread;

        futures.emplace_back(
                pool.enqueue([=] {
                    const float32x4_t v_inv255 = vdupq_n_f32(1.0f / 255.0f);

                    for (int y = start_row; y < end_row; ++y) {
                        const uint8_t* in_row = input_buf + ((crop_y + y) * input_w + crop_x) * 4;
                        float* out_row = output_buf + y * crop_w * 3;

                        int x = 0;

                        for (; x <= crop_w - 8; x += 8) {
                            // 1. 加载8个RGBA像素 (de-interleaving)
                            // 使用 vld4_u8 加载8个像素，而不是 vld4q_u8 加载16个
                            uint8x8x4_t rgba = vld4_u8(in_row + x * 4);

                            // 2. 将uint8拓宽到uint16
                            uint16x8_t r_u16 = vmovl_u8(rgba.val[0]);
                            uint16x8_t g_u16 = vmovl_u8(rgba.val[1]);
                            uint16x8_t b_u16 = vmovl_u8(rgba.val[2]);

                            // 3. 将uint16分为高低两部分，并拓宽到uint32
                            uint32x4_t r_u32_low  = vmovl_u16(vget_low_u16(r_u16));
                            uint32x4_t r_u32_high = vmovl_u16(vget_high_u16(r_u16));
                            uint32x4_t g_u32_low  = vmovl_u16(vget_low_u16(g_u16));
                            uint32x4_t g_u32_high = vmovl_u16(vget_high_u16(g_u16));
                            uint32x4_t b_u32_low  = vmovl_u16(vget_low_u16(b_u16));
                            uint32x4_t b_u32_high = vmovl_u16(vget_high_u16(b_u16));

                            // 4. 转换为float并归一化
                            float32x4x3_t rgb_out_low, rgb_out_high;
                            rgb_out_low.val[0]  = vmulq_f32(vcvtq_f32_u32(r_u32_low), v_inv255);
                            rgb_out_low.val[1]  = vmulq_f32(vcvtq_f32_u32(g_u32_low), v_inv255);
                            rgb_out_low.val[2]  = vmulq_f32(vcvtq_f32_u32(b_u32_low), v_inv255);

                            rgb_out_high.val[0] = vmulq_f32(vcvtq_f32_u32(r_u32_high), v_inv255);
                            rgb_out_high.val[1] = vmulq_f32(vcvtq_f32_u32(g_u32_high), v_inv255);
                            rgb_out_high.val[2] = vmulq_f32(vcvtq_f32_u32(b_u32_high), v_inv255);

                            // 5. 存储为交错的RGB float
                            // 存储前4个像素
                            vst3q_f32(out_row + (x + 0) * 3, rgb_out_low);
                            // 存储后4个像素
                            vst3q_f32(out_row + (x + 4) * 3, rgb_out_high);
                        }

                        // Process remaining pixels in the row
                        for (; x < crop_w; ++x) {
                            int in_idx = x * 4;
                            int out_idx = x * 3;
                            output_buf[out_idx + 0] = in_row[in_idx + 0] * (1.0f / 255.0f);
                            output_buf[out_idx + 1] = in_row[in_idx + 1] * (1.0f / 255.0f);
                            output_buf[out_idx + 2] = in_row[in_idx + 2] * (1.0f / 255.0f);
                        }
                    }
                })
        );
    }

    for (auto& f : futures) {
        f.get();
    }
}


// .\ffmpeg.exe -stream_loop -1 -i G:\Desktop\206.mp4 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
// .\ffmpeg.exe -stream_loop -1 -re -i G:\Desktop\nemo.mp4 -c:v copy -b:v 2000k -framerate 30 -rtsp_transport tcp -f rtsp rtsp://172.18.166.246:8554/mystream
