package com.meti;

import java.util.HashMap;
import java.util.List;
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
    public static final int EPOCH_COUNT = 100;

    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), key -> key % 2 == 0));
        var trainingData = new Data(data);

        var trained = IntStream.range(0, EPOCH_COUNT)
                .boxed()
                .reduce(random(), (network1, integer) -> {
                    var state = trainingData.streamBatches(BATCH_COUNT).reduce(new NetworkState(network1, 0),
                            (network2, batch) -> {
                                var result = network2.network().trainBatch(trainingData, batch);
                                return new NetworkState(result.network(), network2.mse() + result.mse());
                            },
                            StreamUtils::selectRight);

                    System.out.println(integer + "," + (state.mse() / BATCH_COUNT));
                    return state.network();
                }, StreamUtils::selectRight);

        var totalCorrect = trainingData.stream().mapToInt(entry -> {
            var outputVector = trained.forward(trainingData, entry.getKey());
            var evenValue = outputVector.apply(0);
            var oddValue = outputVector.apply(1);

            var isEven = entry.getValue();
            if (isEven && evenValue > oddValue) {
                return 1;
            } else if (!isEven && evenValue < oddValue) {
                return 1;
            } else {
                return 0;
            }
        }).sum();

        var percentage = (double) totalCorrect / trainingData.size();
        System.out.println((percentage * 100) + "%");
    }

    public static Network random() {
        var nodes = new HashMap<Integer, Node>();
        nodes.put(HIDDEN_ID, Node.random(1));
        nodes.put(HIDDEN1_ID, Node.random(1));
        nodes.put(HIDDEN2_ID, Node.random(1));
        nodes.put(EVEN_ID, Node.random(3));
        nodes.put(ODD_ID, Node.random(3));

        var topology = new HashMap<Integer, List<Integer>>();
        topology.put(HIDDEN_ID, List.of(EVEN_ID, ODD_ID));
        topology.put(HIDDEN1_ID, List.of(EVEN_ID, ODD_ID));
        topology.put(HIDDEN2_ID, List.of(EVEN_ID, ODD_ID));

        return new GraphNetwork(new MapNodes(nodes), topology);
    }
}
