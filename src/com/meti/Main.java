package com.meti;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    public static final double LEARNING_RATE = 1d;
    public static final int BATCH_COUNT = 10;
    public static final int EVEN_ID = 3;
    public static final int HIDDEN_ID = 0;
    public static final int ODD_ID = 4;
    public static final int HIDDEN1_ID = 1;
    public static final int HIDDEN2_ID = 2;

    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), key -> key % 2 == 0));

        var network = random();
        System.out.println(network);

        var trained = data.entrySet()
                .stream()
                .collect(Collectors.groupingBy(entry -> entry.getKey() % BATCH_COUNT))
                .values().stream().reduce(network, (network1, batch) -> {
                    var gradientSum = batch.stream().reduce(network1.zero(), (gradientSumNetwork1, entry) -> {
                        var input = normalize(data, entry);
                        var inputVector = Vector.from(input);

                        var results = new HashMap<Integer, Double>();

                        var hiddenValue = network1.apply(HIDDEN_ID).forward(inputVector);
                        var hiddenValue1 = network1.apply(HIDDEN1_ID).forward(inputVector);
                        var hiddenValue2 = network1.apply(HIDDEN2_ID).forward(inputVector);
                        results.put(HIDDEN_ID, hiddenValue);
                        results.put(HIDDEN1_ID, hiddenValue1);
                        results.put(HIDDEN2_ID, hiddenValue2);

                        var evenInputVector = new Vector(network1.streamConnections(0)
                                .map(results::get)
                                .collect(Collectors.toList()));
                        var evenValue = network1.apply(EVEN_ID).forward(evenInputVector);

                        var oddInputVector = new Vector(network1.streamConnections(0)
                                .map(results::get)
                                .collect(Collectors.toList()));

                        var oddValue = network1.apply(ODD_ID).forward(oddInputVector);

                        var isEven = entry.getValue();
                        var expectedEven = isEven ? 1d : 0d;
                        var expectedOdd = isEven ? 0d : 1d;

                        var costDerivative = 2 * (evenValue - expectedEven + oddValue + expectedOdd);

                        var evenActivated = sigmoidDerivative(evenValue);
                        var evenBase = costDerivative * evenActivated;
                        var evenGradient = new Node(evenInputVector, 1d).multiply(evenBase);

                        var oddActivated = sigmoidDerivative(oddValue);
                        var oddBase = costDerivative * oddActivated;
                        var oddGradient = new Node(oddInputVector, 1d).multiply(oddBase);

                        var hiddenActivated = sigmoidDerivative(hiddenValue);
                        var hiddenBase = (evenBase * network.apply(EVEN_ID).weight().apply(0) +
                                          oddBase * network.apply(ODD_ID).weight().apply(0)) * hiddenActivated;

                        var hiddenActivated1 = sigmoidDerivative(hiddenValue);
                        var hiddenBase1 = (evenBase * network.apply(EVEN_ID).weight().apply(1) +
                                           oddBase * network.apply(ODD_ID).weight().apply(1)) * hiddenActivated1;

                        var hiddenActivated2 = sigmoidDerivative(hiddenValue);
                        var hiddenBase2 = (evenBase * network.apply(EVEN_ID).weight().apply(2) +
                                           oddBase * network.apply(ODD_ID).weight().apply(2)) * hiddenActivated2;

                        var hiddenGradient = new Node(inputVector, 1d).multiply(hiddenBase);
                        var hiddenGradient1 = new Node(inputVector, 1d).multiply(hiddenBase1);
                        var hiddenGradient2 = new Node(inputVector, 1d).multiply(hiddenBase2);

                        var network3 = gradientSumNetwork1.addToNode(evenGradient, EVEN_ID);
                        var network2 = network3.addToNode(oddGradient, ODD_ID);
                        var network4 = network2.addToNode(hiddenGradient, HIDDEN_ID);
                        var network5 = network4.addToNode(hiddenGradient1, HIDDEN1_ID);
                        return network5.addToNode(hiddenGradient2, HIDDEN2_ID);
                    }, Main::selectRight);

                    var gradient = gradientSum.divide(data.size());
                    var newNode = network1.subtract(gradient.multiply(LEARNING_RATE));
                    System.out.println(newNode);
                    return newNode;
                }, Main::selectRight);

        var totalCorrect = data.entrySet().stream().mapToInt(entry -> {
            var input = normalize(data, entry);
            var inputVector = Vector.from(input);

            var hiddenValue = trained.apply(HIDDEN_ID).forward(inputVector);
            var hiddenValue1 = trained.apply(HIDDEN1_ID).forward(inputVector);
            var hiddenValue2 = trained.apply(HIDDEN2_ID).forward(inputVector);
            var hiddenVector = Vector.from(hiddenValue, hiddenValue1, hiddenValue2);

            var evenValue = trained.apply(EVEN_ID).forward(hiddenVector);
            var oddValue = trained.apply(ODD_ID).forward(hiddenVector);

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

    public static Network random() {
        var nodes = new HashMap<Integer, Node>();
        nodes.put(HIDDEN_ID, Node.random(1));
        nodes.put(HIDDEN1_ID, Node.random(1));
        nodes.put(HIDDEN2_ID, Node.random(1));
        nodes.put(EVEN_ID, Node.random(3));
        nodes.put(ODD_ID, Node.random(3));

        var topology = new HashMap<Integer, List<Integer>>();
        topology.put(EVEN_ID, List.of(Main.HIDDEN_ID, Main.HIDDEN1_ID, Main.HIDDEN2_ID));
        topology.put(ODD_ID, List.of(Main.HIDDEN_ID, Main.HIDDEN1_ID, Main.HIDDEN2_ID));

        return new GraphNetwork(nodes, topology);
    }
}
