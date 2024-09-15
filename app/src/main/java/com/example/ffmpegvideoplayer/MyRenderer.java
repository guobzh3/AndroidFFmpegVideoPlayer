

package com.example.ffmpegvideoplayer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL;
import javax.microedition.khronos.opengles.GL10;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLUtils;
import android.opengl.Matrix;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.io.OutputStream;
import java.io.FileOutputStream;

import com.example.ffmpegvideoplayer.R;
import com.example.ffmpegvideoplayer.RawResourceReader;
import com.example.ffmpegvideoplayer.ShaderHelper;
import com.example.ffmpegvideoplayer.TextureHelper;
//interface BitmapHandler {
//   void setBitmap(Bitmap bitmap);
//}
public class MyRenderer implements GLSurfaceView.Renderer {
    private String gl_tag = "opengl";
    private String shader_tag = "shader";
    // context（这个是干啥用的）
    private final Context mActivityContext;
    private long start_time , end_time , cost_time ;

    // 世界顶点数据
    final float[] vertex_points;
    final float[] vertex_points_2;

    // 纹理坐标数据
    final float[] fragment_points;
//    final float[] fragment_points_2;

    // 字节数
    private final int floatDataSize = 4;

    // 两个着色器的顶点数量
    private final int position_data_size = 4;
    private final int color_data_size = 4;
    private int width;
    private int height;
    // 程序句柄
    private int program_handle;
    // 着色器句柄
    private int vertex_shader_handle;
    private int fragment_shader_handle;
    // 纹理句柄（GL的函数要用一个数组来接住）
    private int[] mTextureDataHandle = new int[2];
//    private int mTextureDataHandle;
    // position句柄（这一部分要先写了着色器程序才知道）

    private int mPosition_handle;
    // accord句柄
    private int mcord_handle;

    public Bitmap bi_pic;

    public Bitmap sr_pic;

    public boolean updateSurface_Flag = false;

    // 顶点坐标buffer(VBO)
    // 纹理坐标buffer（VBO）

    FloatBuffer vertex_buffer;
    FloatBuffer vertex_buffer_2;
    FloatBuffer fragment_buffer;

    private Bitmap test_bitmap;

    //    FloatBuffer fragment_buffer_2;
//
//    private int texture1_id;
//    private int texture2_id;
    // 构造函数
    public MyRenderer(Context mActivityContext) {
        this.mActivityContext = mActivityContext;

        // 初始化顶点数据
        // 原始的着色器：顶点一个要左上，左下，右上，右下来初始化，不然会崩，我也不知道为啥
        vertex_points = new float[]{
                -1.0f, 1.0f,
                -1.0f, -1.0f,
                1.0f, 1.0f,
                1.0f, -1.0f
        };
//         原始的片元着色器：上下要颠倒过来
        fragment_points = new float[]{
                0.0f, 0.0f,
                0.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, 1.0f
        };

        // 只在中间显示的着色器
//        vertex_points_2 = new float[] {
//                -0.5f , 0.5f ,
//                -0.5f , -0.5f ,
//                0.5f , 0.5f,
//                0.5f , -0.5f
//        };
        // 只在右下显示的着色器
        vertex_points_2 = new float[]{
                0.0f, 0.0f,
                0.0f, -1.0f,
                1.0f, 0.0f,
                1.0f, -1.0f
        };
        // 直接拷贝GLStudio-master 的代码
//        vertex_points = new float[] {
//                -1.0f, -1.0f,
//                -1.0f, 1.0f,
//                1.0f, 1.0f,
//                1.0f, -1.0f
//        };
//        fragment_points = new float[] {
//                0.0f, 1.0f,
//                0.0f, 0.0f,
//                1.0f, 0.0f,
//                1.0f, 1.0f
//        };
        // 画第二张图的坐标
//        vertex_points_2 = new float[] {
//                -0.2f, -0.2f,
//                -0.2f, 0.2f,
//                0.2f, 0.2f,
//                0.2f, -0.2f
//        };
//        fragment_points_2 = vertex_to_texture(vertex_points_2);


        vertex_buffer = ByteBuffer.allocateDirect(vertex_points.length * floatDataSize)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        vertex_buffer.put(vertex_points).position(0); // 记得把数据放进去

        fragment_buffer = ByteBuffer.allocateDirect(fragment_points.length * floatDataSize)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        fragment_buffer.put(fragment_points).position(0); // 记得把数据放进去

        // 第二张图的坐标
        vertex_buffer_2 = ByteBuffer.allocateDirect(vertex_points_2.length * floatDataSize)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        vertex_buffer_2.put(vertex_points_2).position(0);
//
//        // 妈的，这里没有放入数据
//        fragment_buffer_2 = ByteBuffer.allocateDirect(fragment_points_2.length * floatDataSize)
//                .order(ByteOrder.nativeOrder())
//                .asFloatBuffer();
//        fragment_buffer_2.put(fragment_points_2).position(0); // 记得把数据放进去

    }

    protected String getVertexShader() {
        return RawResourceReader.readTextFileFromRawResource(mActivityContext, R.raw.vertex_shader);
    }

    protected String getFragmentShader() {
        return RawResourceReader.readTextFileFromRawResource(mActivityContext, R.raw.fragment_shader);
    }

    @Override
    // 创建上下文，着色器程序，
    public void onSurfaceCreated(GL10 gl10, EGLConfig eglConfig) {
        // Set the background clear color to black.(设置背景颜色：没有被渲染到的位置的颜色）
        GLES20.glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        // Use culling to remove back faces.
        GLES20.glEnable(GLES20.GL_CULL_FACE);

        // Enable depth testing 启用深度测试。深度测试是一种技术，用于确定场景中哪些物体应该被绘制在前面，哪些应该被绘制在后面。当启用深度测试后，OpenGL ES 会根据每个像素的深度值来决定是否绘制该像素。这有助于确保只有前景物体被正确渲染。
        //如果没有这行代码: 默认情况下，深度测试可能未启用。这意味着所有物体都会被绘制，不论其相对于摄像机的深度如何。这会导致渲染结果中的物体出现穿插现象，即物体之间互相穿过，破坏了三维空间中的视觉深度。
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);

        // 创建着色器
        final String vertex_shader_sourcecode = getVertexShader();
        final String fragment_shader_sourcecode = getFragmentShader();
        // 顶点着色器 (创建顶点着色器）
        vertex_shader_handle = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
        Log.i(gl_tag, "vertex shader handle: " + vertex_shader_handle);
        if (vertex_shader_handle != 0) {
            GLES20.glShaderSource(vertex_shader_handle, vertex_shader_sourcecode); // 输入着色器代码
            GLES20.glCompileShader(vertex_shader_handle); // 编译
            Log.i(gl_tag, "vextex shader compile");
            // 检查时候编译成功
            final int[] complie_status = new int[1];
            GLES20.glGetShaderiv(vertex_shader_handle, GLES20.GL_COMPILE_STATUS, complie_status, 0);

            if (complie_status[0] == 0) {
                Log.i(gl_tag, "vextex shader compile error");
//                int shaderHandle;
                Log.e(shader_tag, "Error compiling shader: " + GLES20.glGetShaderInfoLog(vertex_shader_handle));
                GLES20.glDeleteShader(vertex_shader_handle);
                vertex_shader_handle = 0;
            }
        } else {
            throw new RuntimeException("Error creating vertex shader.");
        }

        fragment_shader_handle = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
        Log.i(gl_tag, "fragment shader handle: " + fragment_shader_handle);

        if (fragment_shader_handle != 0) {
            GLES20.glShaderSource(fragment_shader_handle, fragment_shader_sourcecode); // 输入着色器代码
            GLES20.glCompileShader(fragment_shader_handle); // 编译
            Log.i(gl_tag, "fragment shader compile");

            // 检查时候编译成功
            final int[] complie_status = new int[1];
            GLES20.glGetShaderiv(fragment_shader_handle, GLES20.GL_COMPILE_STATUS, complie_status, 0);

            if (complie_status[0] == 0) {
//                int shaderHandle;
                Log.i(gl_tag, "framgent shader compile error");

                Log.e(shader_tag, "Error compiling shader: " + GLES20.glGetShaderInfoLog(fragment_shader_handle));
                GLES20.glDeleteShader(fragment_shader_handle);
                fragment_shader_handle = 0;
            }
        } else {
            throw new RuntimeException("error creating  fragment shader!");
        }

        // 创建；链接shader；bind attribute；链接program
        program_handle = GLES20.glCreateProgram(); // 创建着色器程序
        Log.i(gl_tag, "program handle:" + program_handle);
        if (program_handle != 0) { // 连接两个着色器
            Log.i(gl_tag, "linking: vertex" + vertex_shader_handle + " fragment: " + fragment_shader_handle);
            GLES20.glAttachShader(program_handle, vertex_shader_handle);
            GLES20.glAttachShader(program_handle, fragment_shader_handle);

            // Bind attributes 绑定attribute对象到不同的顶点属性 i（这个是有什么用来着）
//            GLES20.glBindAttribLocation(program_handle , 0 , "a_position");
//            GLES20.glBindAttribLocation(program_handle , 1 , "a_coord");
//            GLES20.glBindAttribLocation(program_handle , 2 , "a_coord2");

            GLES20.glLinkProgram(program_handle); // 链接着色器程序
            // 检查program创建是否成功

            final int[] status = new int[1];
            GLES20.glGetProgramiv(program_handle, GLES20.GL_LINK_STATUS, status, 0);
            if (status[0] == 0) {
                Log.e("program", "Error compiling program: " + GLES20.glGetProgramInfoLog(program_handle));
                GLES20.glDeleteProgram(program_handle);
                program_handle = 0;
            }
        } else {
            throw new RuntimeException("error creating program handles");
        }

        // 我要在这里先创建1个纹理，在后面draw的时候把这两个纹理填充进来就行
        GLES20.glGenTextures(1, mTextureDataHandle, 0); // 生成一个纹理句柄，用于在opengl中标识纹理对象

        if (mTextureDataHandle[0] == 0) {
            throw new RuntimeException("error generating texture name.");
        }
        Log.i(gl_tag, "accord2 handle: " + GLES20.glGetAttribLocation(program_handle, "a_coord2"));
        // 在opengl中绑定资源
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mTextureDataHandle[0]);
//        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D , mTextureDataHandle[1]);

        // 设置缩小和放大使用的算法
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        // 设置超出的时候的填充算法
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_REPEAT);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_REPEAT);

        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;	// No pre-scaling 不进行预缩放（默认情况下，Android会根据设备的分辨率和你放置图片的资源文件目录而预先缩放位图。我们不希望Android根据我们的情况对位图进行缩放，因此我们将inScaled设置为false）

        test_bitmap = BitmapFactory.decodeResource(mActivityContext.getResources(), R.drawable.wechat_pic, options);

//        // 在opengl 中绑定纹理资源
//        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D , mTextureDataHandle[0]);
//
//        // 设置缩小和放大使用的算法
//        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER , GLES20.GL_NEAREST);
//        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D , GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
//
//        // Load the bitmap into the bound texture. 将位图加载到绑定的纹理中（与上文glbindtexture绑定的纹理id相对应 ）
//        // 将像素数据上传到纹理对象
//        // 最后一个参数是偏移量，表示从 Bitmap 数据的开始位置开始读取
//        // 索引 0 表示最高分辨率的纹理层，即原始纹理
        // 把bitmap的数据拷贝到这个纹理中
//        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
//
//        // 回收bitmap
//        bitmap.recycle();
//
//        // 设置第一个id
//        texture1_id = mTextureDataHandle[0];
//        // 导入第二个纹理（这个纹理的helper后面再学习了！）
//        texture2_id = TextureHelper.loadTexture(mActivityContext , R.drawable.usb_android);
    }

    @Override
    // 刷新一下页面就完事
    public void onSurfaceChanged(GL10 gl10, int width, int height) {
        width = 3840;
        height = 2160;
        GLES20.glViewport(0, 0, width, height);
        this.width = width;
        this.height = height;
//        GLES20.glViewport(0, 0, width, height);
    }

    @Override
    // 画图，传入顶点数据等，然后声明适应的glprogrampointer
    public void onDrawFrame(GL10 gl10) {
        if (this.updateSurface_Flag == false){
            return;
        }
        else {
            start_time = System.currentTimeMillis();
            // 获取 attribute 属性的句柄
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
            // 设置使用的着色器程序
            GLES20.glUseProgram(program_handle);
            mPosition_handle = GLES20.glGetAttribLocation(program_handle, "a_position");
            mcord_handle = GLES20.glGetAttribLocation(program_handle, "a_coord");
            int u_Texture_location = GLES20.glGetUniformLocation(program_handle, "u_Texture");

            Log.i(gl_tag, "locations: " + program_handle + " " + mPosition_handle + " " + mcord_handle + " " + u_Texture_location); // 3 0 1 -1 0 -1

            // Set the active texture unit to texture unit 0.
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, this.sr_pic, 0);
            // Bind the texture to this unit. 绑定纹理（这个在前面也绑定过了）
            // 应该是每个纹理有一个纹理id，这里绑定那一个就说明使用的是哪一个
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mTextureDataHandle[0]);
            //        Tell the texture uniform sampler to use this texture in the shader by binding to texture unit 0.
            GLES20.glUniform1i(u_Texture_location, 0); // 纹理单位 0

            // 传递数据到opengl中
            vertex_buffer_2.position(0);
            GLES20.glVertexAttribPointer(mPosition_handle, 2, GLES20.GL_FLOAT, false, 0, vertex_buffer_2);
            GLES20.glEnableVertexAttribArray(mPosition_handle);

            fragment_buffer.position(0);
            GLES20.glVertexAttribPointer(mcord_handle, 2, GLES20.GL_FLOAT, false, 0, fragment_buffer);
            GLES20.glEnableVertexAttribArray(mcord_handle);
            //
            ////         根据当前的顶点进行绘制
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
            //        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, 4  );
            Log.i(gl_tag, "draw sr ");

            // 画第二个图
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, this.bi_pic, 0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mTextureDataHandle[0]);
            GLES20.glUniform1i(u_Texture_location, 0); // 纹理单位 1
            ////
            vertex_buffer.position(0);
            GLES20.glVertexAttribPointer(mPosition_handle, 2, GLES20.GL_FLOAT, false, 0, vertex_buffer);
            GLES20.glEnableVertexAttribArray(mPosition_handle);
            ////
            fragment_buffer.position(0);
            GLES20.glVertexAttribPointer(mcord_handle, 2, GLES20.GL_FLOAT, false, 0, fragment_buffer);
            GLES20.glEnableVertexAttribArray(mcord_handle);

            //
            //        GLES20.glActiveTexture(GLES20.GL_TEXTURE1); //激活第二个纹理——激活纹理单元意味着接下来的所有纹理绑定操作都将作用于这个纹理单元。
            //        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D , texture2_id); // 绑定纹理到激活的纹理中
            //        GLES20.glUniform1i(u_Texture_location_2 , 1); // 设置texture2的值为纹理单元1
            //
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
            //        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, 4 );
            end_time = System.currentTimeMillis();
            cost_time = end_time - start_time;
            Log.i("time" , "gl render time: " + cost_time + " ms");
            Log.i(gl_tag, "draw bi");
            readBufferPixelToBitmap(this.width,this.height);
//            this.updateSurface_Flag = false;
        }
    }

    public void set_bi_Bitmap(Bitmap pic) { // 对应mtextureHandle[1]
        this.bi_pic = pic;
    }

    public void set_sr_bitmap(Bitmap pic) { // 对应 mTextureHandle[0]
        this.sr_pic = pic;
    }

    public void readBufferPixelToBitmap(int width, int height) {
        ByteBuffer buf = ByteBuffer.allocateDirect(width * height * 4);
        buf.order(ByteOrder.nativeOrder());
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf);
        buf.rewind();

        bmp.copyPixelsFromBuffer(buf);
        Log.i("save_img" , Environment.getExternalStorageDirectory().getAbsolutePath());
//        String dir = Environment.getExternalStorageDirectory().getAbsolutePath() + "/save_rendered_frames";
        //获取内部存储状态
        String state = Environment.getExternalStorageState();
        //如果状态不是mounted，无法读写
        if (!state.equals(Environment.MEDIA_MOUNTED)) {
            Log.i("save_img","state error");
            return ;
        }
        else {
            Log.i("save_img" , "state good");
        }
//        保存文件
//        String fileName = "test_save_4k_image";
        try {
            File file = new File(Environment.getExternalStorageDirectory()+"/test_save_4k_image.jpg");
            FileOutputStream out = new FileOutputStream(file);
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
            Log.i("save_img","success save");
        } catch (Exception e) {
            Log.i("save_img","save_error");

            e.printStackTrace();
        }

//        return bmp;
    }

}