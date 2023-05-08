package com.meti;

public record Node(Vector weight, double bias) {
    public Node(double bias) {
        this(Vector.zero(0), bias);
    }

    @Override
    public String toString() {
        return weight.toString() + "," + bias;
    }

    public static Node zero(int size) {
        return new Node(Vector.zero(size), 0);
    }

    public static Node random(int size) {
        return new Node(Vector.random(size), Math.random());
    }

    public Node zero() {
        return new Node(Vector.zero(weight.size()), 0);
    }

    public double forward(Vector input) {
        if (weight.size() != input.size()) {
            var format = "Weight size '%d' did not match input vector size '%d'.";
            var message = format.formatted(weight.size(), input.size());
            throw new IllegalArgumentException(message);
        }

        var evaluated = weight.multiply(input).sum() + bias;
        return NetMath.sigmoid(evaluated);
    }

    public Node multiply(double scalar) {
        return new Node(weight.multiply(scalar), bias * scalar);
    }

    public Node add(Node other) {
        return new Node(weight.add(other.weight), bias + other.bias);
    }

    public Node subtract(Node other) {
        return new Node(weight.add(other.weight.negate()), bias - other.bias);
    }
}
