package com.meti;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
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
                        var inputVector = Vector.from(input);

                        var hiddenValue = network1.hidden.forward(inputVector);
                        var hiddenValue1 = network1.hidden1.forward(inputVector);
                        var hiddenValue2 = network1.hidden1.forward(inputVector);
                        var hiddenVector = Vector.from(hiddenValue, hiddenValue1, hiddenValue2);

                        var evenValue = network1.even.forward(hiddenVector);
                        var oddValue = network1.odd.forward(hiddenVector);

                        var isEven = entry.getValue();
                        var expectedEven = isEven ? 1d : 0d;
                        var expectedOdd = isEven ? 0d : 1d;

                        var costDerivative = 2 * (evenValue - expectedEven + oddValue + expectedOdd);

                        var evenActivated = sigmoidDerivative(evenValue);
                        var evenBase = costDerivative * evenActivated;
                        var evenGradient = new Node(hiddenVector, 1d).multiply(evenBase);

                        var oddActivated = sigmoidDerivative(oddValue);
                        var oddBase = costDerivative * oddActivated;
                        var oddGradient = new Node(hiddenVector, 1d).multiply(oddBase);

                        var hiddenActivated = sigmoidDerivative(hiddenValue);
                        var hiddenBase = (evenBase * network.even.weight.apply(0) +
                                          oddBase * network.odd.weight.apply(0)) * hiddenActivated;

                        var hiddenActivated1 = sigmoidDerivative(hiddenValue);
                        var hiddenBase1 = (evenBase * network.even.weight.apply(1) +
                                           oddBase * network.odd.weight.apply(1)) * hiddenActivated1;

                        var hiddenActivated2 = sigmoidDerivative(hiddenValue);
                        var hiddenBase2 = (evenBase * network.even.weight.apply(2) +
                                           oddBase * network.odd.weight.apply(2)) * hiddenActivated2;

                        var hiddenGradient = new Node(inputVector, 1d).multiply(hiddenBase);
                        var hiddenGradient1 = new Node(inputVector, 1d).multiply(hiddenBase1);
                        var hiddenGradient2 = new Node(inputVector, 1d).multiply(hiddenBase2);

                        return gradientSumNetwork1
                                .addEven(evenGradient)
                                .addOdd(oddGradient)
                                .addHidden(hiddenGradient)
                                .addHidden1(hiddenGradient1)
                                .addHidden2(hiddenGradient2);
                    }, Main::selectRight);

                    var gradient = gradientSum.divide(data.size());
                    var newNode = network1.subtract(gradient.multiply(LEARNING_RATE));
                    System.out.println(newNode);
                    return newNode;
                }, Main::selectRight);

        var totalCorrect = data.entrySet().stream().mapToInt(entry -> {
            var input = normalize(data, entry);
            var inputVector = Vector.from(input);

            var hiddenValue = trained.hidden.forward(inputVector);
            var hiddenValue1 = trained.hidden1.forward(inputVector);
            var hiddenValue2 = trained.hidden1.forward(inputVector);
            var hiddenVector = Vector.from(hiddenValue, hiddenValue1, hiddenValue2);

            var evenValue = trained.even.forward(hiddenVector);
            var oddValue = trained.odd.forward(hiddenVector);

            var isEven = entry.getValue();
            if (isEven && evenValue > oddValue) {
                return 1;
            } else if (!isEven && evenValue < oddValue) {
                return 1;
            } else {
                return 0;
            }
        }).sum();

        var percentage = (double) totalCorrect / (double) data.size();
        System.out.println((percentage * 100) + "%");
    }

    private static Network zero() {
        return new Network(
                Node.zero(1), Node.zero(1), Node.zero(1),
                Node.zero(3), Node.zero(3));
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

    private record Vector(List<Double> values) {

        public static Vector zero(int size) {
            return supply(size, index -> 0d);
        }

        private static Vector supply(int size, IntFunction<Double> supplier) {
            return new Vector(IntStream.range(0, size)
                    .mapToObj(supplier)
                    .toList());
        }

        public static Vector random(int size) {
            return supply(size, index -> Math.random());
        }

        public static Vector from(double... values) {
            return new Vector(DoubleStream.of(values).boxed().collect(Collectors.toList()));
        }

        private double apply(int index) {
            return values.get(index);
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

        private int size() {
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
    }

    private record Node(Vector weight, double bias) {
        public static Node zero(int size) {
            return new Node(Vector.zero(size), 0);
        }

        private static Node random(int size) {
            return new Node(Vector.random(size), Math.random());
        }

        private double forward(Vector input) {
            if (weight.size() != input.size()) {
                var format = "Weight size '%d' did not match input vector size '%d'.";
                var message = format.formatted(weight.size(), input.size());
                throw new IllegalArgumentException(message);
            }

            var evaluated = weight.multiply(input).sum() + bias;
            return 1d / (1d + Math.pow(Math.E, -evaluated));
        }

        public Node multiply(double scalar) {
            return new Node(weight.multiply(scalar), bias * scalar);
        }

        public Node add(Node other) {
            return new Node(weight.add(other.weight), bias + other.bias);
        }

        public Node divide(double scalar) {
            return multiply(1d / scalar);
        }

        public Node subtract(Node other) {
            return new Node(weight.add(other.weight.negate()), bias - other.bias);
        }
    }

    private record Network(Node hidden, Node hidden1, Node hidden2, Node even, Node odd) {
        private static Network random() {
            return new Network(Node.random(1), Node.random(1), Node.random(1),
                    Node.random(3), Node.random(3));
        }

        private Network addEven(Node outputGradient) {
            return new Network(hidden, hidden1, hidden2, even.add(outputGradient), odd);
        }

        private Network addHidden(Node node) {
            return new Network(hidden.add(node), hidden1, hidden2, even, odd);
        }

        public Network divide(double scalar) {
            return new Network(hidden.divide(scalar), hidden1.divide(scalar), hidden2.divide(scalar),
                    even.divide(scalar), odd.divide(scalar));
        }

        public Network multiply(double scalar) {
            return new Network(hidden.multiply(scalar), hidden1.multiply(scalar), hidden1.multiply(scalar),
                    even.divide(scalar), odd.divide(scalar));
        }

        public Network subtract(Network other) {
            return new Network(hidden.subtract(other.hidden), hidden1.subtract(other.hidden1), hidden2.subtract(other.hidden2),
                    even.subtract(other.even), odd.subtract(other.odd));
        }

        public Network addOdd(Node odd) {
            return new Network(hidden, hidden1, hidden2, even, odd);
        }

        public Network addHidden1(Node hidden1) {
            return new Network(hidden, hidden1.add(hidden1), hidden2, even, odd);
        }

        public Network addHidden2(Node hidden2) {
            return new Network(hidden, hidden1, hidden2.add(hidden2), even, odd);
        }
    }
}
