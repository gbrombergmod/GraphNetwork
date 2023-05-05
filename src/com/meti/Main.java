package com.meti;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    public static final double LEARNING_RATE = 1;
    public static final int CHUNK_COUNT = 10;

    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), value -> value % 2 == 0));

        var chunks = data.entrySet()
                .stream()
                .collect(Collectors.groupingBy(entry -> entry.getKey() % CHUNK_COUNT))
                .values();

        var trained = chunks.stream().reduce(random(), (Network network, List<Map.Entry<Integer, Boolean>> entries) -> {
            var gradientSum = data.entrySet().stream().reduce(zero(), (gradientSumNetwork, entry) -> {
                var normalized = normalize(data, entry);
                var hiddenActivated = forward(network.hidden, normalized);
                var outputActivated = forward(network.output, hiddenActivated);

                var isEven = entry.getValue();
                var expected = isEven ? 0d : 1d;
                var cost = 2d * (outputActivated - expected);

                var outputActivatedDerivative = sigmoidDerivative(outputActivated);
                var outputBaseDerivative = cost * outputActivatedDerivative;
                var outputGradient = new Node(hiddenActivated, 1d).multiply(outputBaseDerivative);
                var network1 = gradientSumNetwork.withOutputGradient(outputGradient);

                var hiddenActivatedDerivative = sigmoidDerivative(hiddenActivated);
                var hiddenBaseDerivative = outputBaseDerivative * network.output.weight() * hiddenActivatedDerivative;
                var hiddenGradient = new Node(normalized, 1d).multiply(hiddenBaseDerivative);

                return network1.withHiddenGradient(hiddenGradient);
            }, Main::chooseRight);

            System.out.println(network);

            return network.subtract(gradientSum.divide(data.size()).multiply(LEARNING_RATE));
        }, Main::chooseRight);

        var totalCorrect = data.entrySet().stream().mapToInt(entry -> {
            var normalized = normalize(data, entry);
            var hiddenActivated = forward(trained.hidden, normalized);
            var outputActivated = forward(trained.output, hiddenActivated);

            var isEven = entry.getValue();
            if (isEven && outputActivated < 0.5) {
                return 1;
            } else if (!isEven && outputActivated >= 0.5) {
                return 1;
            } else {
                return 0;
            }
        }).sum();

        System.out.println(totalCorrect);
    }

    private static Network zero() {
        return new Network(Node.zero(), Node.zero());
    }

    private static double sigmoidDerivative(double outputActivated) {
        return outputActivated * (1 - outputActivated);
    }

    private static Network random() {
        return new Network(Node.random(), Node.random());
    }

    private static double normalize(Map<Integer, Boolean> data, Map.Entry<Integer, Boolean> entry) {
        var input = entry.getKey();
        return (double) input / (double) data.size();
    }

    private static double forward(Node node, double input) {
        var value = node.weight() * input + node.bias();
        return 1d / (1d + Math.pow(Math.E, -value));
    }

    private static <T> T chooseRight(T left, T right) {
        return right;
    }

    record Network(Node hidden, Node output) {

        private Network withOutputGradient(Node gradient) {
            return new Network(hidden, output.copy().add(gradient));
        }

        private Network withHiddenGradient(Node gradient) {
            return new Network(hidden.copy().add(gradient), output);
        }

        public Network divide(double scalar) {
            return new Network(hidden.divide(scalar), output.divide(scalar));
        }

        public Network multiply(double scalar) {
            return new Network(hidden.multiply(scalar), output.multiply(scalar));
        }

        public Network subtract(Network other) {
            return new Network(hidden.subtract(other.hidden), output.subtract(other.output));
        }
    }
}