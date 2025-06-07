package com.example.ffmpegvideoplayer;

public class TaggedData<T> {
    public final long frameId;
    public final T data;

    public TaggedData(long frameId, T data) {
        this.frameId = frameId;
        this.data = data;
    }
}