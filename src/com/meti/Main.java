package com.meti;

import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {
    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), key -> key % 2 == 0));

        var weight = Math.random();
        var bias = Math.random();
        var node = new Node(weight, bias);
        System.out.println(node);

        var gradientSum = data.entrySet().stream().reduce(Node.zero(), (gradientSum1, entry) -> {
            var key = (double) entry.getKey();
            var input = key / (double) data.size();
            var evaluated = node.weight * input + node.bias;
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
        var newNode = node.subtract(gradient);
        System.out.println(newNode);
    }

    private record Node(double weight, double bias) {
        public static Node zero() {
            return new Node(0, 0);
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
