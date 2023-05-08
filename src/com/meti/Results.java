package com.meti;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public record Results(Map<Integer, Double> cache) {
    public Results() {
        this(new HashMap<>());
    }

    Results insert(int id, double result) {
        cache.put(id, result);
        return this;
    }

    Vector locate(List<Integer> ids) {
        return new Vector(ids
                .stream()
                .map(cache::get)
                .collect(Collectors.toList()));
    }

    public double apply(int id) {
        return cache.get(id);
    }
}