package com.meti;

public record Node(double weight, double bias) {
    public static Node zero() {
        return new Node(0, 0);
    }

    static Node random() {
        var weight = Math.random();
        var bias = Math.random();
        return new Node(weight, bias);
    }

    public Node multiply(double scalar) {
        return new Node(weight * scalar, bias * scalar);
    }

    public Node subtract(Node other) {
        return new Node(weight - other.weight, bias - other.bias);
    }

    public Node add(Node other) {
        return new Node(weight + other.weight, bias + other.bias);
    }

    public Node divide(double scalar) {
        return new Node(weight / scalar, bias / scalar);
    }
}