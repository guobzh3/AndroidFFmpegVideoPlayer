package com.example.ffmpegvideoplayer.analysis;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import com.example.ffmpegvideoplayer.MinimalEGLContext;
import com.qualcomm.qti.QnnDelegate;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.RuntimeFlavor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class InferenceTFLite {

    private static final String TAG = "[Inference TFLite]";
    private final String MODEL_FILE = "quicsr_ds_dStride2_270pto540p_noresnet.tflite";
    private final Size INPNUT_SIZE = new Size(480, 270); // Width, Height
    private final int[] OUTPUT_SIZE = new int[]{1, 540, 960, 1}; // N, H, W, C
    private final Boolean IS_INT8 = true; // 用于量化路径的用户定义标志

    private Interpreter tflite;
    private final Interpreter.Options options; // 所有代理都将配置到这个实例上
    private ImageProcessor imageProcessor;

    // 量化参数
    private final MetadataExtractor.QuantizationParams input5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 0);

    // EGL 上下文，按原样保留
    private MinimalEGLContext eglCtx = new MinimalEGLContext();
    private boolean eglSetupSuccessful = eglCtx.setup();

    public InferenceTFLite() {
        // 在构造函数中初始化 Options 对象，确保它在所有代理方法中都可用
        this.options = new Interpreter.Options();
        Log.d(TAG, "InferenceTFLite instance created with a new Interpreter.Options object.");
    }

    public void initialModel(Context context) {
        if (eglSetupSuccessful) {
            Log.i(TAG, "EGL Context setup successful.");
        } else {
            Log.w(TAG, "EGL Context setup failed.");
        }

        try {
            // 使用配置好的 this.options 来创建 Interpreter
            ByteBuffer tfliteModel = FileUtil.loadMappedFile(context, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, this.options);

            // 根据是否量化来配置图像处理器
            if (IS_INT8) {
                imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0f, 255f))
                        .add(new org.tensorflow.lite.support.common.ops.QuantizeOp(input5SINT8QuantParams.getZeroPoint(), input5SINT8QuantParams.getScale()))
                        .add(new CastOp(DataType.UINT8))
                        .build();
            } else {
                imageProcessor = new ImageProcessor.Builder()
                        // .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0f, 255f))
                        .build();
            }
            Log.i(TAG, "Success loading model: " + MODEL_FILE);

        } catch (Exception e) {
            Log.e(TAG, "Error initializing TFLite interpreter or model: " + MODEL_FILE, e);
            Toast.makeText(context, "Model initialization error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    public int[] getOUTPUT_SIZE() {
        return OUTPUT_SIZE;
    }

    public TensorBuffer superResolution(TensorImage modelInput, int[] tf_output_shape) {
        // 你的原始代码将输出形状解释为 [N, W, H, C]，这里保持该行为
        int[] bufferShape = new int[]{1, tf_output_shape[1], tf_output_shape[0], 3};

        if (tflite == null) {
            Log.e(TAG, "TFLite interpreter is not initialized. Cannot run inference.");
            // 返回一个空的或默认的 TensorBuffer
            return TensorBuffer.createFixedSize(bufferShape, IS_INT8 ? DataType.UINT8 : DataType.FLOAT32);
        }

        DataType outputDataType = tflite.getOutputTensor(0).dataType();
        TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(bufferShape, outputDataType);

        // 运行推理
        tflite.run(modelInput.getBuffer(), outputTensorBuffer.getBuffer());

        return outputTensorBuffer;
    }

    public void addNNApiDelegate() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            try {
                NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
                nnApiOptions.setAllowFp16(true);
                nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);

                NnApiDelegate nnApiDelegate = new NnApiDelegate(nnApiOptions);
                this.options.addDelegate(nnApiDelegate); // 修改类的成员 'options'
                Log.i(TAG, "NNAPI delegate added successfully.");
            } catch (Exception e) {
                Log.w(TAG, "Failed to create NNAPI delegate, falling back to CPU.", e);
            }
        } else {
            Log.w(TAG, "NNAPI is not supported on this device (API level < 28).");
        }
    }

    /**
     * 【已重构】尝试添加 GPU 代理。
     * 如果失败，会自动回退到使用 CPU 线程。
     * 使用了最新的 TFLite API。
     */
    public void addGPUDelegate() {
        try {
            // 1. 创建 GPU 代理的配置选项
            GpuDelegateFactory.Options delegateOptions = new GpuDelegateFactory.Options();
            delegateOptions.setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);

            delegateOptions.setQuantizedModelsAllowed(true);


            // (调试技巧) 如果遇到问题，可以强制指定后端进行测试
             delegateOptions.setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENCL);

            // 2. 使用工厂模式创建 GpuDelegate
            // 【核心改动】将代理选项传入构造函数，并调用无参数的 create()
            GpuDelegate gpuDelegate = (GpuDelegate) new GpuDelegateFactory(delegateOptions).create(RuntimeFlavor.APPLICATION);

            // 3. 将代理添加到类的成员 'options' 中
            this.options.addDelegate(gpuDelegate);
            Log.i(TAG, "GPU delegate added successfully.");

        } catch (Exception e) {
            Log.w(TAG, "GPU delegate is not supported or failed to initialize. Falling back to CPU.", e);
            // 如果添加 GPU 代理失败，则回退到使用 CPU 线程
            addThread(4);
        }
    }

    public void addThread(int threadCount) {
        this.options.setNumThreads(threadCount);
        Log.i(TAG, "CPU threads set to: " + threadCount);
    }

    public void addQNNDelegate(Activity activity) {
        try {
            QnnDelegate.Options qnnOptions = new QnnDelegate.Options();
            qnnOptions.setSkelLibraryDir(activity.getApplicationInfo().nativeLibraryDir);
            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.GPU_BACKEND);
            qnnOptions.setGpuPerformanceMode(QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_HIGH);
            qnnOptions.setGpuPrecision(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_HYBRID);

            QnnDelegate qnnDelegate = new QnnDelegate(qnnOptions);
            this.options.addDelegate(qnnDelegate); // 修改类的成员 'options'
            Log.i(TAG, "Qualcomm QNN Delegate added with backend: " + qnnOptions.getBackendType());

        } catch (Exception e) {
            Log.e(TAG, "Failed to create or add QNN Delegate. TFLite will use CPU or other delegates.", e);
            Toast.makeText(activity, "QNN Delegate not loaded: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
            Log.i(TAG, "TFLite interpreter closed.");
        }
        // Interpreter.close() 会自动关闭所有已添加的代理，无需手动关闭
    }
}