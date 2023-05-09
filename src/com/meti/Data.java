package com.meti;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record Data<T>(Map<Integer, T> data) {
    Stream<List<Map.Entry<Integer, T>>> streamBatches(int batchCount) {
        var itemsPerBatch = data.size() / batchCount;
        return stream()
                .collect(Collectors.groupingBy(entry -> entry.getKey() / itemsPerBatch))
                .values()
                .stream();
    }

    Stream<Map.Entry<Integer, T>> stream() {
        return data().entrySet().stream();
    }

    double normalize(int key) {
        return key / 1000d;
    }

    public double size() {
        return data.size();
    }
}