package com.example.ffmpegvideoplayer.analysis;

import android.content.Context;
import android.graphics.Bitmap; // 为完整性保留，尽管未直接用于此重构的核心逻辑
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor; // 为完整性保留
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor; // 为完整性保留
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
// import org.tensorflow.lite.support.common.ops.DequantizeOp; // 原始代码中未使用，保留供参考
// import org.tensorflow.lite.support.common.ops.QuantizeOp; // 在 ImageProcessor 中使用

import com.qualcomm.qti.QnnDelegate;
import android.app.Activity;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays; // For Arrays.equals

public class InferenceTFLite {
    private Interpreter tflite;
    Interpreter.Options options = new Interpreter.Options();
    private final Size INPNUT_SIZE = new Size(480, 270); // Width, Height

    // Model file names (original selection logic preserved)
    //    private String MODEL_FILE = "quicsr_270p.tflite";
    //    private String MODEL_FILE = "quicsr_int32_270pto540p.tflite";
    //    private String MODEL_FILE = "quicsr_v2_270pto540p.tflite";
    //    private String MODEL_FILE = "quicsr_ds_270pto540p.tflite";
    //    private String MODEL_FILE = "quicsr_ds_540pto1080p.tflite";
    private String MODEL_FILE = "quicsr_ds_dStride2_270pto540p_noresnet.tflite";
    //    private String MODEL_FILE = "quicsr_ds_dStride2_540pto1080p_1.tflite";
    //    private String MODEL_FILE = "quicsr_v2_270pto540p_matrix.tflite";
    //    private String MODEL_FILE = "quicsr_v2_270pto540p_matmul.tflite";

    // 此常量由 getOUTPUT_SIZE() 返回
    // 原始: {1, 540, 960, 1} (NHWC, 1 通道)
    private final int[] OUTPUT_SIZE = new int[]{1, 540, 960, 1};

    // 适用于 270p -> 1080p 模型 (原始选择逻辑保留)
    //    private String MODEL_FILE = "quicsr_float32_270pto1080p.tflite";
    //    private String MODEL_FILE = "quicsr_int32_270pto1080p.tflite";
    //    private String MODEL_FILE = "quicsr_ds_270pto1080p.tflite";
    //    private final int[] OUTPUT_SIZE = new int[] {1, 1080, 1920, 3}; // NHWC, 3 channels

    private static final String TAG = "[Inference TFLite]";
    private Boolean IS_INT8 = false; // 用于量化路径的用户定义标志

    // Quantization params (original values preserved)
    MetadataExtractor.QuantizationParams input5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 0);
    // output5SINT8QuantParams 在原始提供的代码中未用于 superResolution 的返回值。
    // MetadataExtractor.QuantizationParams output5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 0);

    ImageProcessor imageProcessor;

    // 用于保存可重用输出 TensorBuffer 的成员变量
    private TensorBuffer hwcOutputTensorBuffer;
    // 用于跟踪创建 hwcOutputTensorBuffer 的形状，以处理动态 tf_output_shape
    private int[] currentTfOutputShapeForBuffer = null;


    public void initialModel(Context activity) { // 参数名 'activity' 保持不变，尽管它是一个 Context
        try {
            // 要tflite 2.16.1 版本才支持 Transpose version 6操作 (Original comment)
            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);

            // hwcOutputTensorBuffer 不再在此处创建。它将在 superResolution 中按需创建。

            if (IS_INT8) {
                imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0f, 255f)) // 为清晰起见，明确为浮点型
                        .add(new org.tensorflow.lite.support.common.ops.QuantizeOp(input5SINT8QuantParams.getZeroPoint(), input5SINT8QuantParams.getScale()))
                        .add(new CastOp(DataType.UINT8))
                        .build();
            } else {
                imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0f, 255f)) // 为清晰起见，明确为浮点型
                        .build();
            }
            Log.i(TAG, "Success loading model: " + MODEL_FILE);
        } catch (IOException e) {
            Log.e(TAG, "Error loading model: " + MODEL_FILE, e);
            Toast.makeText(activity, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        } catch (Exception e) { // 捕获其他初始化错误的通用异常
            Log.e(TAG, "Error initializing model: " + MODEL_FILE, e);
            Toast.makeText(activity, "Model initialization error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    public int[] getOUTPUT_SIZE() {
        return OUTPUT_SIZE; // 返回类常量 {1, 540, 960, 1}
    }

    // 这里超分直接返回的是tensorBuffer，并没有进行后续的转换bitmap等操作 (Original comment)
    public TensorBuffer superResolution(TensorImage modelInput, int[] tf_output_shape) {
        // tf_output_shape 假定为超分辨率图像的 [高度, 宽度]。
        // 原始代码将 output_size 构造为: {1, tf_output_shape[1], tf_output_shape[0], 3}
        // This implies an NWHC format: [Batch, Width, Height, Channels=3]
        // 我们将坚持这种解释以保持原始行为。
        int[] bufferShape = new int[]{1, tf_output_shape[1], tf_output_shape[0], 3};

        if (tflite == null) {
            Log.e(TAG, "TFLite interpreter is null. Cannot run superResolution.");
            // Attempt to return a buffer consistent with previous logic if it was ever created,
            // 或者如果在此错误状态下调用（尽管它不会被填充），则创建一个新的。
            if (hwcOutputTensorBuffer == null || !Arrays.equals(this.currentTfOutputShapeForBuffer, tf_output_shape)) {
                Log.w(TAG, "Creating a fallback output buffer because TFLite is null. Shape: " + Arrays.toString(bufferShape));
                // 如果 tflite 对象不可用以获取实际数据类型，则回退到 FLOAT32
                this.hwcOutputTensorBuffer = TensorBuffer.createFixedSize(bufferShape, DataType.FLOAT32);
                this.currentTfOutputShapeForBuffer = tf_output_shape.clone(); // tf_output_shape 是 [高,宽]
            }
            return hwcOutputTensorBuffer;
        }

        // 检查输出缓冲区是否需要由于形状更改或首次运行而（重新）创建
        if (hwcOutputTensorBuffer == null || !Arrays.equals(this.currentTfOutputShapeForBuffer, tf_output_shape)) {
            DataType outputDataType = tflite.getOutputTensor(0).dataType();
            // int[] actualModelOutputShape = tflite.getOutputTensor(0).shape();
            // Log.d(TAG, "Actual model output shape from TFLite: " + Arrays.toString(actualModelOutputShape));
            // Log.d(TAG, "Creating buffer with user-provided tf_output_shape interpreted as NWHC: " + Arrays.toString(bufferShape));

            // 确保 bufferShape 尽可能与模型的实际输出尺寸匹配，
            // 但原始代码将通道硬编码为 3，并使用 tf_output_shape 作为高/宽。
            // 我们优先匹配原始代码的缓冲区构造逻辑。
            this.hwcOutputTensorBuffer = TensorBuffer.createFixedSize(bufferShape, outputDataType);
            this.currentTfOutputShapeForBuffer = tf_output_shape.clone(); // 存储 [高,宽] 对
            Log.i(TAG, "Created/Recreated hwcOutputTensorBuffer with shape: " +
                    Arrays.toString(bufferShape) + " and DataType: " + outputDataType);
        }

        // 确保在 tflite.run() 之前重置 ByteBuffer
        // 输入缓冲区
        ByteBuffer inputBuffer = modelInput.getBuffer();
        inputBuffer.rewind();

        // 输出缓冲区
        ByteBuffer outputBuffer = hwcOutputTensorBuffer.getBuffer();
        outputBuffer.rewind(); // Reset position to 0
        // outputBuffer.limit(outputBuffer.capacity()); // Ensure limit is capacity, though createFixedSize should handle this.

        tflite.run(inputBuffer, outputBuffer);

        return hwcOutputTensorBuffer;
    }


    public void addNNApiDelegate() {
        NnApiDelegate nnApiDelegate = null;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
            nnApiOptions.setAllowFp16(true); // 如果可用且有益，则允许 FP16 精度
            //ANEURALNETWORKS_PREFER_LOW_POWER：倾向于以最大限度减少电池消耗的方式执行。这种设置适合经常执行的编译。
            //ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER：倾向于尽快返回单个答案，即使这会耗费更多电量。这是默认值。
            //ANEURALNETWORKS_PREFER_SUSTAINED_SPEED：倾向于最大限度地提高连续帧的吞吐量，例如，在处理来自相机的连续帧时。
            nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
            // nnApiOptions.setUseNnapiCpu(true); // Generally not recommended unless specific need
            // nnApiOptions.setCacheDir(...); // Optional: for compilation caching
            // nnApiOptions.setModelToken(...); // Optional: for compilation caching
            nnApiDelegate = new NnApiDelegate(nnApiOptions);
            options.addDelegate(nnApiDelegate);
            Log.i(TAG, "NNAPI delegate added with custom options.");
        } else {
            Log.w(TAG, "NNAPI delegate not supported on this API level. Falling back to CPU threads if configured.");
            // 如果 NNAPI 不可用，则回退或替代方案（例如，确保设置了线程）
            // addThread(4); // Original code had this as an alternative path
        }
    }

    // 2.8.0 tensorflow 才能用这个代理，2.16.1用不了 (Original comment)
    public void addGPUDelegate() {
        CompatibilityList compatibilityList = new CompatibilityList();
        if (compatibilityList.isDelegateSupportedOnThisDevice()) {
            GpuDelegate.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
            Log.i(TAG, "GPU delegate added.");
        } else {
            Log.w(TAG, "GPU delegate is not supported on this device. Falling back to CPU threads if configured.");
            // addThread(4); // Original code had this as an alternative path
        }
    }

    public void addThread(int thread) {
        options.setNumThreads(thread);
        Log.i(TAG, "Set number of TFLite threads to: " + thread);
    }



    public void addQNNDelegate(Activity activity) { // 参数是 Activity
//            try {
//                System.loadLibrary("cdsprpc");
//                Log.i(TAG, "Successfully loaded libcdsprpc.so");
//            } catch (UnsatisfiedLinkError e) {
//                Log.e(TAG, "Failed to load libcdsprpc.so: " + e.getMessage(), e);
//            }

        try {
            QnnDelegate.Options qnnOptions = new QnnDelegate.Options();
            // 原始 QNN 选项保留
             qnnOptions.setBackendType(QnnDelegate.Options.BackendType.GPU_BACKEND);
             qnnOptions.setGpuPerformanceMode(QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_HIGH);
             qnnOptions.setGpuPrecision(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16);
//            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND);
//            qnnOptions.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_HIGH_PERFORMANCE);
//            qnnOptions.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16);
//            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.DSP_BACKEND);
//            qnnOptions.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN);

            if (activity != null && activity.getApplicationInfo() != null) {
                qnnOptions.setSkelLibraryDir(activity.getApplicationInfo().nativeLibraryDir);
            } else {
                Log.w(TAG, "Activity or ApplicationInfo is null for SkelLibraryDir. HTP/DSP backend might fail for QNN Delegate.");
            }
            // qnnOptions.setLogLevel(QnnDelegate.Options.LogLevel.INFO); // Debugging

            QnnDelegate qnnDelegate = new QnnDelegate(qnnOptions);
            this.options.addDelegate(qnnDelegate); // Add to the class member 'options'
            Log.i(TAG, "Qualcomm QNN Delegate added with backend: " + qnnOptions.getBackendType());

        } catch (Exception e) { // Catching generic Exception as QNN might throw various errors
            Log.e(TAG, "Failed to create or add QNN Delegate: " + e.getMessage(), e);
            Toast.makeText(activity, "QNN Delegate not loaded: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            // 如果 QNN 失败，TFLite 将在 CPU 或其他可用代理上运行。
        }
    }

    // 可选：关闭解释器并释放资源的方法
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
            Log.i(TAG, "TFLite interpreter closed.");
        }
        // 添加到选项的代理由 Interpreter 实例管理。
        // 如果代理本身需要显式关闭（例如，QnnDelegate），
        // 它们应该作为成员存储并在此处关闭。
        // NnApiDelegate 和 GpuDelegate 通常不需要在 Interpreter 之外显式关闭。
        // QnnDelegate 可能会，如果它作为成员存储，请检查其文档。
        // 在此代码中，代理是其 addDelegate 方法的本地代理或匿名代理，
        // 因此 Interpreter.close() 应该处理它们的生命周期。
        // 如果 QnnDelegate 已存储：
        // if (qnnDelegateMember != null) {
        // qnnDelegateMember.close();
        // qnnDelegateMember = null;
        // }
    }
}