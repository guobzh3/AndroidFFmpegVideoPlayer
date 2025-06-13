package com.example.ffmpegvideoplayer;

import android.util.Log;

import java.nio.ByteBuffer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class SharedByteBuffer {

    private static final int POOL_CAPACITY = 16; // Match the queue capacity
    private static final BlockingQueue<SharedByteBuffer> pool = new ArrayBlockingQueue<>(POOL_CAPACITY);

    public final ByteBuffer buffer;
    private final AtomicInteger refCount = new AtomicInteger(0);

    private SharedByteBuffer(int capacity) {
        this.buffer = ByteBuffer.allocateDirect(capacity);
    }

    public static void initialize(int bufferCapacity) {
        for (int i = 0; i < POOL_CAPACITY; i++) {
            pool.offer(new SharedByteBuffer(bufferCapacity));
        }
    }

    public static SharedByteBuffer obtain() {
        try {
            // Always block until a buffer is available. This simplifies logic and backpressure.
            return pool.take();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }
    public void addRef() {
            addRef(1);
        }

    public void addRef(int val) {
        refCount.addAndGet(val);
    }



    public void release() {
        if (refCount.decrementAndGet() == 0) {
            // Reset buffer position for next use
            buffer.clear();
            pool.offer(this);
        }
        if (pool.size() < 8) Log.i("buffer", "Buffer released, pool capicity: " +  pool.size());
    }
}