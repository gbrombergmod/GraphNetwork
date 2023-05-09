package com.meti;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public interface Data<T> {
    Stream<List<Map.Entry<Integer, T>>> streamBatches(int batchCount);

    Stream<Map.Entry<Integer, T>> stream();

    double normalize(int key);

    double size();
}
