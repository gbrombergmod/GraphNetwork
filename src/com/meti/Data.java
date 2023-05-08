package com.meti;

import java.util.Map;

public record Data(Map<Integer, Boolean> data) {
    double normalize(int key) {
        var castedKey = (double) key;
        return castedKey / (double) data.size();
    }

    public double size() {
        return data.size();
    }
}