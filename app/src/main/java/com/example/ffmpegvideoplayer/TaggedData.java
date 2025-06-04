package com.example.ffmpegvideoplayer;

public class TaggedData<T> {
    private T data;
    private long frameNumber;

    // No-argument constructor for pooling
    public TaggedData() {
        this.data = null;
        this.frameNumber = -1; // Default invalid frame number
    }

    public TaggedData(T data, long frameNumber) {
        this.data = data;
        this.frameNumber = frameNumber;
    }

    public void set(T data, long frameNumber) {
        this.data = data;
        this.frameNumber = frameNumber;
    }

    public T getData() {
        return data;
    }

    public long getFrameNumber() {
        return frameNumber;
    }

    // Method to reset fields when returning to a pool
    public void reset() {
        this.data = null;
        this.frameNumber = -1;
    }
}