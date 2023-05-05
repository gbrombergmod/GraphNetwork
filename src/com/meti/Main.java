package com.meti;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    public static final double LEARNING_RATE = 0.0001;

    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), value -> value % 2 == 0));

        var weight = Math.random();
        var bias = Math.random();
        var node = new Node(weight, bias);

        var reduce = data.entrySet().stream().reduce(node, (node1, entry) -> {
            var input = entry.getKey();
            var normalized = (double) input / (double) data.size();
            var activated = forward(node1, normalized);

            var isEven = entry.getValue();
            var expected = isEven ? 0d : 1d;
            var cost = 2d * (activated - expected);
            var activatedDerivative = activated * (1 - activated);
            var baseDerivative = cost * activatedDerivative;
            var gradient = new Node(input, 1d).multiply(baseDerivative);
            return node1.subtract(gradient.multiply(LEARNING_RATE));
        }, (node12, node2) -> node2);

        var totalCorrect = data.entrySet().stream().mapToInt(entry -> {
            var input = entry.getKey();
            var normalized = (double) input / (double) data.size();
            var activated = forward(reduce, normalized);

            var isEven = entry.getValue();
            if (isEven && activated < 0.5) {
                return 1;
            } else if (!isEven && activated >= 0.5) {
                return 1;
            } else {
                return 0;
            }
        }).sum();

        System.out.println(totalCorrect);
    }

    private static double forward(Node node, double input) {
        var value = node.weight() * input + node.bias();
        return 1d / (1d + Math.pow(Math.E, -value));
    }
}