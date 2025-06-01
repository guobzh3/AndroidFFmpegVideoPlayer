package com.example.ffmpegvideoplayer.analysis;

import android.content.Context;
import android.graphics.Bitmap; // Kept for completeness, though not directly used in this refactor's core logic
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor; // Kept for completeness
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor; // Kept for completeness
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
// import org.tensorflow.lite.support.common.ops.DequantizeOp; // Not used in original, kept for reference
// import org.tensorflow.lite.support.common.ops.QuantizeOp; // Used in ImageProcessor

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

    // This constant is returned by getOUTPUT_SIZE()
    // Original: {1, 540, 960, 1} (NHWC, 1 channel)
    private final int[] OUTPUT_SIZE = new int[]{1, 540, 960, 1};

    // For 270p -> 1080p models (original selection logic preserved)
    //    private String MODEL_FILE = "quicsr_float32_270pto1080p.tflite";
    //    private String MODEL_FILE = "quicsr_int32_270pto1080p.tflite";
    //    private String MODEL_FILE = "quicsr_ds_270pto1080p.tflite";
    //    private final int[] OUTPUT_SIZE = new int[] {1, 1080, 1920, 3}; // NHWC, 3 channels

    private static final String TAG = "[Inference TFLite]";
    private Boolean IS_INT8 = false; // User-defined flag for quantization path

    // Quantization params (original values preserved)
    MetadataExtractor.QuantizationParams input5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 0);
    // output5SINT8QuantParams was unused in the original provided code for superResolution's return.
    // MetadataExtractor.QuantizationParams output5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 0);

    ImageProcessor imageProcessor;

    // Member variable to hold the reusable output TensorBuffer
    private TensorBuffer hwcOutputTensorBuffer;
    // To track the shape for which hwcOutputTensorBuffer was created, to handle dynamic tf_output_shape
    private int[] currentTfOutputShapeForBuffer = null;


    public void initialModel(Context activity) { // Parameter name 'activity' kept as is, though it's a Context
        try {
            // 要tflite 2.16.1 版本才支持 Transpose version 6操作 (Original comment)
            ByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, MODEL_FILE);
            tflite = new Interpreter(tfliteModel, options);

            // hwcOutputTensorBuffer is no longer created here. It will be created on-demand in superResolution.

            if (IS_INT8) {
                imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0f, 255f)) // Explicitly float for clarity
                        .add(new org.tensorflow.lite.support.common.ops.QuantizeOp(input5SINT8QuantParams.getZeroPoint(), input5SINT8QuantParams.getScale()))
                        .add(new CastOp(DataType.UINT8))
                        .build();
            } else {
                imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0f, 255f)) // Explicitly float for clarity
                        .build();
            }
            Log.i(TAG, "Success loading model: " + MODEL_FILE);
        } catch (IOException e) {
            Log.e(TAG, "Error loading model: " + MODEL_FILE, e);
            Toast.makeText(activity, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        } catch (Exception e) { // Catch generic Exception for other initialization errors
            Log.e(TAG, "Error initializing model: " + MODEL_FILE, e);
            Toast.makeText(activity, "Model initialization error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    public int[] getOUTPUT_SIZE() {
        return OUTPUT_SIZE; // Returns the class constant {1, 540, 960, 1}
    }

    // 这里超分直接返回的是tensorBuffer，并没有进行后续的转换bitmap等操作 (Original comment)
    public TensorBuffer superResolution(TensorImage modelInput, int[] tf_output_shape) {
        // tf_output_shape is assumed to be [Height, Width] of the super-resolved image.
        // Original code constructed output_size as: {1, tf_output_shape[1], tf_output_shape[0], 3}
        // This implies an NWHC format: [Batch, Width, Height, Channels=3]
        // We will stick to this interpretation to maintain original behavior.
        int[] bufferShape = new int[]{1, tf_output_shape[1], tf_output_shape[0], 3};

        if (tflite == null) {
            Log.e(TAG, "TFLite interpreter is null. Cannot run superResolution.");
            // Attempt to return a buffer consistent with previous logic if it was ever created,
            // or create a new one if called in this error state (though it won't be filled).
            if (hwcOutputTensorBuffer == null || !Arrays.equals(this.currentTfOutputShapeForBuffer, tf_output_shape)) {
                Log.w(TAG, "Creating a fallback output buffer because TFLite is null. Shape: " + Arrays.toString(bufferShape));
                // Fallback to FLOAT32 if tflite object is not available to get actual data type
                this.hwcOutputTensorBuffer = TensorBuffer.createFixedSize(bufferShape, DataType.FLOAT32);
                this.currentTfOutputShapeForBuffer = tf_output_shape.clone(); // tf_output_shape is [H,W]
            }
            return hwcOutputTensorBuffer;
        }

        // Check if the output buffer needs to be (re)created due to shape change or first run
        if (hwcOutputTensorBuffer == null || !Arrays.equals(this.currentTfOutputShapeForBuffer, tf_output_shape)) {
            DataType outputDataType = tflite.getOutputTensor(0).dataType();
            // int[] actualModelOutputShape = tflite.getOutputTensor(0).shape();
            // Log.d(TAG, "Actual model output shape from TFLite: " + Arrays.toString(actualModelOutputShape));
            // Log.d(TAG, "Creating buffer with user-provided tf_output_shape interpreted as NWHC: " + Arrays.toString(bufferShape));

            // Ensure the bufferShape matches the model's actual output dimensions if possible,
            // but the original code hardcoded channels to 3 and used tf_output_shape for H/W.
            // We prioritize matching the original code's buffer construction logic.
            this.hwcOutputTensorBuffer = TensorBuffer.createFixedSize(bufferShape, outputDataType);
            this.currentTfOutputShapeForBuffer = tf_output_shape.clone(); // Store the [H,W] pair
            Log.i(TAG, "Created/Recreated hwcOutputTensorBuffer with shape: " +
                    Arrays.toString(bufferShape) + " and DataType: " + outputDataType);
        }

        // Ensure ByteBuffers are reset before tflite.run()
        // Input buffer
        ByteBuffer inputBuffer = modelInput.getBuffer();
        inputBuffer.rewind();

        // Output buffer
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
            nnApiOptions.setAllowFp16(true); // Allow FP16 precision if available and beneficial
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
            // Fallback or alternative if NNAPI not available (e.g., ensure threads are set)
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



    public void addQNNDelegate(Activity activity) { // Parameter is Activity
            try {
                System.loadLibrary("cdsprpc");
                Log.i(TAG, "Successfully loaded libcdsprpc.so");
            } catch (UnsatisfiedLinkError e) {
                Log.e(TAG, "Failed to load libcdsprpc.so: " + e.getMessage(), e);
            }

        try {
            QnnDelegate.Options qnnOptions = new QnnDelegate.Options();
            // Original QNN options preserved
//             qnnOptions.setBackendType(QnnDelegate.Options.BackendType.GPU_BACKEND);
//             qnnOptions.setGpuPerformanceMode(QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_HIGH);
//             qnnOptions.setGpuPrecision(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16);
            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND);
            qnnOptions.setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_HIGH_PERFORMANCE);
            qnnOptions.setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16);
//            qnnOptions.setBackendType(QnnDelegate.Options.BackendType.DSP_BACKEND);
            qnnOptions.setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN);

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
            // If QNN fails, TFLite will run on CPU or other available delegates.
        }
    }

    // Optional: Method to close the interpreter and release resources
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
            Log.i(TAG, "TFLite interpreter closed.");
        }
        // Delegates added to options are managed by the Interpreter instance.
        // If delegates themselves need explicit closing (e.g., QnnDelegate),
        // they should be stored as members and closed here.
        // NnApiDelegate and GpuDelegate typically don't require explicit closing beyond the Interpreter.
        // QnnDelegate might, check its documentation if it's stored as a member.
        // In this code, delegates are local to their addDelegate methods or anonymous,
        // so Interpreter.close() should handle their lifecycle.
        // If QnnDelegate was stored:
        // if (qnnDelegateMember != null) {
        // qnnDelegateMember.close();
        // qnnDelegateMember = null;
        // }
    }
}