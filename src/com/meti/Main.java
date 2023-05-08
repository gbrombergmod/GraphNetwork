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

        var network = TempNetwork.random();
        System.out.println(network);

        var trained = data.entrySet()
                .stream()
                .collect(Collectors.groupingBy(entry -> entry.getKey() % BATCH_COUNT))
                .values().stream().reduce(network, (network1, batch) -> {
                    var gradientSum = batch.stream().reduce(network1.zero(), (gradientSumNetwork1, entry) -> {
                        var input = normalize(data, entry);
                        var inputVector = Vector.from(input);

                        var hiddenValue = network1.getHidden().forward(inputVector);
                        var hiddenValue1 = network1.getHidden1().forward(inputVector);
                        var hiddenValue2 = network1.getHidden1().forward(inputVector);
                        var hiddenVector = Vector.from(hiddenValue, hiddenValue1, hiddenValue2);

                        var evenValue = network1.getEven().forward(hiddenVector);
                        var oddValue = network1.getOdd().forward(hiddenVector);

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
                        var hiddenBase = (evenBase * network.getEven().weight().apply(0) +
                                          oddBase * network.getOdd().weight().apply(0)) * hiddenActivated;

                        var hiddenActivated1 = sigmoidDerivative(hiddenValue);
                        var hiddenBase1 = (evenBase * network.getEven().weight().apply(1) +
                                           oddBase * network.getOdd().weight().apply(1)) * hiddenActivated1;

                        var hiddenActivated2 = sigmoidDerivative(hiddenValue);
                        var hiddenBase2 = (evenBase * network.getEven().weight().apply(2) +
                                           oddBase * network.getOdd().weight().apply(2)) * hiddenActivated2;

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

            var hiddenValue = trained.getHidden().forward(inputVector);
            var hiddenValue1 = trained.getHidden1().forward(inputVector);
            var hiddenValue2 = trained.getHidden1().forward(inputVector);
            var hiddenVector = Vector.from(hiddenValue, hiddenValue1, hiddenValue2);

            var evenValue = trained.getEven().forward(hiddenVector);
            var oddValue = trained.getOdd().forward(hiddenVector);

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

}
