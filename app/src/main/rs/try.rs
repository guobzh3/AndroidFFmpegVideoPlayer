#pragma version(1)
#pragma rs java_package_name(com.example.ffmpegvideoplayer)

// 输入图像
rs_allocation inImage;
// 输出图像
rs_allocation outImage;
// 模糊半径
float blurRadius;

// 定义模糊函数
// v_in ,v_out 是输入输出的数据，usrdata 是传递给内核的任意用户数据，uint32_t x, uint32_t y：是内核的当前执行位置（例如，像素坐标）
void blur(const uchar4 *v_in, uchar4 *v_out, const void *usrData, uint32_t x, uint32_t y) {
    // 计算模糊后的颜色值
    float4 sum = 0.0f;
    int count = 0;
    for (float dx = -blurRadius; dx <= blurRadius; dx++) {
        for (float dy = -blurRadius; dy <= blurRadius; dy++) {
            int newX = x + (int)dx;
            int newY = y + (int)dy;
            if (newX >= 0 && newX < rsAllocationGetDimX(inImage) && newY >= 0 && newY < rsAllocationGetDimY(inImage)) {
                // tips： renderScript 中有内置的函数和api -> 推理感觉rsUnpackColor8888就是内置的函数和api
                sum += rsUnpackColor8888(*v_in + rsAllocationGetElementPtr(inImage, newX, newY));

                count++;
            }
        }
    }
    *v_out = rsPackColor8888(sum / count);
}

// 根函数，RenderScript执行时的入口点
void root() {
    // 获取输入和输出图像的指针
    const uchar4 *in = rsGetAllocationAddress(inImage);
    uchar4 *out = rsGetAllocationAddress(outImage);

    // 执行模糊操作
    blur(in, out, NULL, 0, 0);
}