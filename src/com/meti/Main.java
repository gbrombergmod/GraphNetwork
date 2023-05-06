package com.meti;

import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    public static final double LEARNING_RATE = 0.1d;
    public static final int BATCH_COUNT = 10;

    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), key -> key % 2 == 0));

        var node = Node.random();
        System.out.println(node);

        data.entrySet()
                .stream()
                .collect(Collectors.groupingBy(entry -> entry.getKey() % BATCH_COUNT))
                .values().stream().reduce(node, (node1, batch) -> {
                    var gradientSum = batch.stream().reduce(Node.zero(), (gradientSum1, entry) -> {
                        var key = (double) entry.getKey();
                        var input = key / (double) data.size();
                        var evaluated = node1.weight * input + node1.bias;
                        var activated = 1d / (1d + Math.pow(Math.E, -evaluated));

                        var isEven = entry.getValue();
                        var expected = isEven ? 0d : 1d;
                        var costDerivative = 2 * (activated - expected);
                        var activatedDerivative = activated * (1 - activated);
                        var baseDerivative = costDerivative * activatedDerivative;
                        var gradient = new Node(input, 1d).multiply(baseDerivative);
                        return gradientSum1.add(gradient);
                    }, (previous, next) -> next);

                    var gradient = gradientSum.divide(data.size());
                    var newNode = node1.subtract(gradient.multiply(LEARNING_RATE));
                    System.out.println(newNode);
                    return newNode;
                }, (BinaryOperator<Node>) (node12, node2) -> node2);
    }

    private record Node(double weight, double bias) {
        public static Node zero() {
            return new Node(0, 0);
        }

        private static Node random() {
            var weight = Math.random();
            var bias = Math.random();
            return new Node(weight, bias);
        }

        public Node multiply(double scalar) {
            return new Node(weight * scalar, bias * scalar);
        }

        public Node add(Node other) {
            return new Node(weight + other.weight, bias + other.bias);
        }

        public Node divide(double scalar) {
            return multiply(1d / scalar);
        }

        public Node subtract(Node other) {
            return new Node(weight - other.weight, bias - other.bias);
        }
    }
}
