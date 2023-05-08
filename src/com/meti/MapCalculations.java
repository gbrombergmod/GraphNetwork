package com.meti;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public record MapCalculations(Map<Integer, Double> cache) implements Calculations {
    private MapCalculations() {
        this(new HashMap<>());
    }

    public static Calculations empty() {
        return new MapCalculations();
    }

    @Override
    public Calculations insert(int id, double result) {
        cache.put(id, result);
        return this;
    }

    @Override
    public Vector locate(List<Integer> ids) {
        return new Vector(ids
                .stream()
                .map(cache::get)
                .collect(Collectors.toList()));
    }

    @Override
    public double locate(int id) {
        return cache.get(id);
    }
}