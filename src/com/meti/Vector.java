package com.meti;

import java.util.List;
import java.util.Objects;
import java.util.function.DoubleFunction;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

final class Vector {
    private final List<Double> values;

    Vector(List<Double> values) {
        this.values = values;
    }

    public static Vector zero(int size) {
        return supply(size, index -> 0d);
    }

    private static Vector supply(int size, IntFunction<Double> supplier) {
        return new Vector(IntStream.range(0, size)
                .mapToObj(supplier)
                .toList());
    }

    public static Vector random(int size) {
        return supply(size, index -> NetMath.randomFloat());
    }

    public static Vector from(double... values) {
        return new Vector(DoubleStream.of(values).boxed().collect(Collectors.toList()));
    }

    public double apply(int index) {
        if (index < 0 || index >= values.size()) {
            var format = "Index '%d' is out of bounds for size '%d'.";
            var message = format.formatted(index, values.size());
            throw new IndexOutOfBoundsException(message);
        }

        var aDouble = values.get(index);
        if (aDouble == null) {
            throw new IllegalStateException();
        }
        return aDouble;
    }

    public Vector multiply(Vector other) {
        var thisSize = size();
        var otherSize = other.size();
        if (thisSize != otherSize) {
            var format = "Expected a size of '%d' but found '%d'.";
            var message = format.formatted(thisSize, otherSize);
            throw new IllegalArgumentException(message);
        }

        return supply(values.size(), index -> values.get(index) * other.apply(index));
    }

    public int size() {
        return values.size();
    }

    public double sum() {
        return values.stream()
                .mapToDouble(value -> value)
                .sum();
    }

    public Vector multiply(double scalar) {
        return supply(values.size(), index -> values.get(index) * scalar);
    }

    public Vector add(Vector vector) {
        return supply(values.size(), index -> values.get(index) + vector.apply(index));
    }

    public Vector negate() {
        return supply(values.size(), index -> -values.get(index));
    }

    public Vector subtract(Vector other) {
        return add(other.negate());
    }

    public List<Double> values() {
        return values;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (Vector) obj;
        return Objects.equals(this.values, that.values);
    }

    @Override
    public int hashCode() {
        return Objects.hash(values);
    }

    @Override
    public String toString() {
        return values.stream()
                .map(Object::toString)
                .collect(Collectors.joining(","));
    }

    public Vector map(DoubleFunction<Double> mapper) {
        return supply(values().size(), index -> mapper.apply(values.get(index)));
    }
}
