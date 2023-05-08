package com.meti;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public record Data(Map<Integer, Boolean> data) {
    Stream<List<Map.Entry<Integer, Boolean>>> streamBatches(int batchCount) {
        return stream()
                .collect(Collectors.groupingBy(entry -> entry.getKey() % batchCount))
                .values()
                .stream();
    }

    Stream<Map.Entry<Integer, Boolean>> stream() {
        return data().entrySet().stream();
    }

    double normalize(int key) {
        var castedKey = (double) key;
        return castedKey / (double) data.size();
    }

    public double size() {
        return data.size();
    }
}