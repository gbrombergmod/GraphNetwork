package com.meti;

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    public static final double LEARNING_RATE = 1d;
    public static final int BATCH_COUNT = 10;

    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), key -> key % 2 == 0));

        var network = Network.random();
        System.out.println(network);

        var trained = data.entrySet()
                .stream()
                .collect(Collectors.groupingBy(entry -> entry.getKey() % BATCH_COUNT))
                .values().stream().reduce(network, (network1, batch) -> {
                    var gradientSum = batch.stream().reduce(zero(), (gradientSumNetwork1, entry) -> {
                        var input = normalize(data, entry);
                        var hiddenValue = network1.hidden.forward(input);
                        var outputValue = network1.output.forward(hiddenValue);

                        var isEven = entry.getValue();
                        var expected = isEven ? 0d : 1d;
                        var costDerivative = 2 * (outputValue - expected);

                        var outputActivated = sigmoidDerivative(outputValue);
                        var outputBase = costDerivative * okutputActivated;
                        var outputGradient = new Node(hiddenValue, 1d).multiply(outputBase);

                        var hiddenActivated = sigmoidDerivative(hiddenValue);
                        var hiddenBase = outputBase * network.output.weight * hiddenActivated;
                        var hiddenGradient = new Node(input, 1d).multiply(hiddenBase);

                        return gradientSumNetwork1
                                .getWithOutput(outputGradient)
                                .withHidden(hiddenGradient);
                    }, Main::selectRight);

                    var gradient = gradientSum.divide(data.size());
                    var newNode = network1.subtract(gradient.multiply(LEARNING_RATE));
                    System.out.println(newNode);
                    return newNode;
                }, Main::selectRight);

        var totalCorrect = data.entrySet().stream().mapToInt(entry -> {
            var input = normalize(data, entry);
            var result = trained.output.forward(input);
            var isEven = entry.getValue();
            if (isEven && result < 0.5) {
                return 1;
            } else if (!isEven && result >= 0.5) {
                return 1;
            } else {
                return 0;
            }
        }).sum();

        var percentage = (double) totalCorrect / (double) data.size();
        System.out.println((percentage * 100) + "%");
    }

    private static Network zero() {
        return new Network(Node.zero(), Node.zero());
    }

    private static double sigmoidDerivative(double hiddenValue) {
        return hiddenValue * (1 - hiddenValue);
    }

    private static double normalize(Map<Integer, Boolean> data, Map.Entry<Integer, Boolean> entry) {
        var key = (double) entry.getKey();
        return key / (double) data.size();
    }

    private static <T> T selectRight(T t, T t1) {
        return t1;
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

        private double forward(double input) {
            var evaluated = this.weight * input + this.bias;
            return 1d / (1d + Math.pow(Math.E, -evaluated));
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

    private record Network(Node hidden, Node output) {
        private static Network random() {
            var hidden = Node.random();
            var node = Node.random();
            return new Network(hidden, node);
        }

        private Network getWithOutput(Node outputGradient) {
            var newOutput = this.output.add(outputGradient);
            return new Network(this.hidden, newOutput);
        }

        private Network withHidden(Node node) {
            var newHidden = hidden.add(node);
            return new Network(newHidden, output);
        }

        public Network divide(double scalar) {
            return new Network(hidden.divide(scalar), output.divide(scalar));
        }

        public Network multiply(double scalar) {
            return new Network(hidden.multiply(scalar), output.divide(scalar));
        }

        public Network subtract(Network other) {
            return new Network(hidden.subtract(other.hidden), output.subtract(other.output));
        }
    }
}
