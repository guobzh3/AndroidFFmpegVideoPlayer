package com.example.ffmpegvideoplayer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLUtils;

public class TextureHelper
{
    public static int loadTexture(final Context context, final int resourceId)
    {
        final int[] textureHandle = new int[1];
        // 用于生成一个或多个纹理对象的名称。这些名称（也称为纹理句柄）可以用来标识和操作纹理对象
        // 第一个参数 (1): 指定要生成的纹理对象的数量。在这个例子中，我们只生成一个纹理对象。
        // 第二个参数 (textureHandle): 这是一个整型数组的引用，用来存储生成的纹理对象的名字。纹理名字会存放在数组的第一个元素中，因为我们在第一个参数指定了只生成一个纹理对象。
        // 第三个参数 (0): 这个参数指定数组中开始存放纹理名字的位置。在这个例子中，纹理名字将被存放在数组的索引0位置
        // 可以通过gltdeletetexture 来显示地删除纹理
        GLES20.glGenTextures(1, textureHandle, 0); // 生成一个纹理句柄，用于在opengl中标识纹理对象

        if (textureHandle[0] == 0)
        {
            throw new RuntimeException("Error generating texture name.");
        }

        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;	// No pre-scaling 不进行预缩放（默认情况下，Android会根据设备的分辨率和你放置图片的资源文件目录而预先缩放位图。我们不希望Android根据我们的情况对位图进行缩放，因此我们将inScaled设置为false）

        // Read in the resource
        final Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), resourceId, options);

        // Bind to the texture in OpenGL 在OPENGL中绑定纹理资源
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureHandle[0]);

        // Set filtering 设置过滤
        // 设置当纹理被缩小（即从远处观察纹理时）时所使用的过滤方式 filter
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        // 当纹理被放大（即从近处观察纹理时）时所使用的过滤方式
        // 如果采用双线性插值的话，使用GL_LINEAR 参数
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

        // Load the bitmap into the bound texture. 将位图加载到绑定的纹理中（与上文glbindtexture绑定的纹理id相对应 ）
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);

        // Recycle the bitmap, since its data has been loaded into OpenGL. 回收位图，因为数据已经加载到opengl中
        bitmap.recycle();

        return textureHandle[0]; // 返回纹理id：根据这个纹理id就可以操作了
    }
}
