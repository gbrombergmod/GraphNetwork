package com.meti;

import java.util.HashMap;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    public static final double LEARNING_RATE = 1d;
    public static final int BATCH_COUNT = 10;
    public static final int ODD_ID = 4;
    public static final int EPOCH_COUNT = 100;

    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), key -> key * key));

        var trainingData = new Data<>(data);
        var network = random();
        measure(trainingData, network);

        var trained = IntStream.range(0, EPOCH_COUNT)
                .boxed()
                .reduce(network, (network1, integer) -> {
                    var state = trainingData.streamBatches(BATCH_COUNT).reduce(new NetworkState(network1, 0),
                            (network2, batch) -> {
                                var result = network2.network().trainBatch(trainingData, batch, Main::computeExpected);
                                return new NetworkState(result.network(), network2.mse() + result.mse());
                            },
                            StreamUtils::selectRight);
                    return state.network();
                }, StreamUtils::selectRight);

        measure(trainingData, trained);
    }

    private static void measure(Data<Integer> trainingData, Network trained) {
        var totalCorrect = trainingData.stream().mapToInt(entry -> {
            var outputVector = trained.forward(trainingData, entry.getKey());
            var actual = outputVector.apply(0);
            var expected = entry.getValue();

            if (Math.abs(((int) actual) - expected) < 10) {
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
        var topology = new HashMap<Integer, List<Integer>>();

        return getGraphNetwork(new GraphNetworkBuilder(nodes, topology));
    }

    private static GraphNetwork getGraphNetwork(GraphNetworkBuilder graphNetworkBuilder) {
        var hiddenLayer = createLayer(graphNetworkBuilder, 3, 1);
        var hiddenLayer1 = createLayer(graphNetworkBuilder, 3, 3);
        var hiddenLayer2 = createLayer(graphNetworkBuilder, 3, 3);
        var outputLayer = createLayer(graphNetworkBuilder, 1, 3);

        graphNetworkBuilder.connect(hiddenLayer, hiddenLayer1);
        graphNetworkBuilder.connect(hiddenLayer1, hiddenLayer2);
        graphNetworkBuilder.connect(hiddenLayer2, outputLayer);

        return graphNetworkBuilder.toNetwork();
    }

    private static List<Integer> createLayer(GraphNetworkBuilder graphNetworkBuilder, int layerSize, int inputSize) {
        return IntStream.range(0, layerSize)
                .mapToObj(value -> {
                    return graphNetworkBuilder.create(inputSize);
                }).collect(Collectors.toList());
    }

    static Vector computeExpected(int value) {
        return Vector.from(value);
    }
}
