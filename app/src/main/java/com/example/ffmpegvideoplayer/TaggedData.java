package com.example.ffmpegvideoplayer;

public class TaggedData<T> {
    private T data;
    private long frameNumber;

    public TaggedData(T data, long frameNumber) {
        this.data = data;
        this.frameNumber = frameNumber;
    }

    public T getData() {
        return data;
    }

    public long getFrameNumber() {
        return frameNumber;
    }
}