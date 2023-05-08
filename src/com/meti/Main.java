package com.meti;

import java.util.*;
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

                        var collect = network1.computeByDepthsForward()
                                .stream()
                                .flatMap(Collection::stream)
                                .toList();

                        var results = collect.stream().reduce(MapCalculations.empty(),
                                (calculations1, id) -> forward(network1, inputVector, calculations1, id),
                                (previous, next) -> next);

                        var isEven = entry.getValue();
                        var expectedEven = isEven ? 1d : 0d;
                        var expectedOdd = isEven ? 0d : 1d;

                        var actual = results.locate(List.of(EVEN_ID, ODD_ID));
                        var expected = Vector.from(expectedEven, expectedOdd);
                        var costDerivative = 2 * (expected.subtract(actual).sum());

                        var copy = new ArrayList<>(collect);
                        Collections.reverse(copy);

                        var gradientSum1 = collect.stream()
                                .reduce(new Gradients(),
                                        (gradients, integer) -> backwards(gradients, integer, network1, inputVector, results, costDerivative),
                                        (previous, next) -> next);

                        return network1.add(gradientSum1.nodeGradients());
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

    private static Gradients backwards(Gradients gradients, int id, Network network, Vector inputVector, Calculations results, double costDerivative) {
        if (network.isRoot(id)) {
            return backwardsHidden(results, id, gradients, inputVector, network);
        } else {
            var ids = network.listConnections(id);
            var previousInputs = results.locate(ids);
            return backwardsOutput(results, id, gradients, previousInputs, costDerivative);
        }
    }

    private static Gradients backwardsHidden(Calculations results, int source, Gradients gradientSum, Vector inputs, Network network) {
        var previousDerivative = network.listConnections(source)
                .stream()
                .mapToDouble(destination -> gradientSum.baseGradients().locate(destination) * network.findWeight(source, destination))
                .sum();

        return backwardsOutput(results, source, gradientSum, inputs, previousDerivative);
    }

    private static Gradients backwardsOutput(Calculations results, int id, Gradients gradientSum, Vector inputs, double costDerivative) {
        var activated = sigmoidDerivative(results.locate(id));
        var base = costDerivative * activated;
        var gradient = new Node(inputs, 1d).multiply(base);
        return gradientSum.add(id, costDerivative * activated, gradient);
    }

    private static Calculations forward(Network network, Vector inputVector, Calculations calculations1, Integer id) {
        Vector layer;
        if (network.isRoot(id)) {
            layer = inputVector;
        } else {
            var collect = network.listConnections(id);
            layer = calculations1.locate(collect);
        }

        var forward = network.apply(id).forward(layer);
        return calculations1.insert(id, forward);
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

        return new GraphNetwork(new MapNodes(nodes), topology);
    }
}
