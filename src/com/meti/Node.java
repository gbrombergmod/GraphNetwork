package com.meti;

public record Node(double weight, double bias) {
    public Node multiply(double scalar) {
        return new Node(weight * scalar, bias * scalar);
    }

    public Node subtract(Node other) {
        return new Node(weight - other.weight, bias - other.bias);
    }
}