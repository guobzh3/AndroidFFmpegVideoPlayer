package com.example.ffmpegvideoplayer;

import java.nio.ByteBuffer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class SharedByteBuffer {

    private static final int POOL_CAPACITY = 16; // Same as other queues
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
        SharedByteBuffer sbb = pool.poll();
        if (sbb == null) {
            // Pool is empty, this should ideally not happen if pool size is managed well.
            // For robustness, we can create a new one, but this will break the pooling mechanism.
            // Or we can block until one is available.
            // For now, let's assume the pool is large enough.
            // A more robust solution might use take() to block.
            try {
                return pool.take();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
        return sbb;
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
    }
}