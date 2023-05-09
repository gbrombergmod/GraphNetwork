package com.meti;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {

    public static final double LEARNING_RATE = 1d;
    public static final int BATCH_COUNT = 100;
    public static final int ODD_ID = 4;
    public static final int EPOCH_COUNT = 100;

    public static void main(String[] args) throws IOException {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), key -> key));

        var trainingData = new MapData<>(data);
        var network = (Network) build();
        var before = measure(trainingData, network);

        var builder = new StringBuilder();
        builder.append("EPOCH,MSE,");
        var collect = network.stream().toList();
        for (int i = 0; i < collect.size(); i++) {
            Integer nodeID = collect.get(i);
            var prefix = "NODE" + nodeID;
            var node = network.apply(nodeID);
            var weightCount = node.weight().size();

            IntStream.range(0, weightCount).forEach(weightIndex -> builder
                    .append(prefix)
                    .append("_WEIGHT")
                    .append(weightIndex)
                    .append(","));

            builder.append(prefix).append("_BIAS");
            if (i != collect.size() - 1) {
                builder.append(",");
            }
        }
        builder.append("\n");

        var trained = IntStream.range(0, EPOCH_COUNT)
                .boxed()
                .reduce(network, (network1, integer) -> {
                    System.out.println("EPOCH: " + integer);

                    var state = trainingData.streamBatches(BATCH_COUNT).reduce(new NetworkState(network1, 0),
                            (network2, batch) -> {
                                var result = network2.network().trainBatch(trainingData, batch, Main::computeExpected);
                                return new NetworkState(result.network(), network2.mse() + result.mse());
                            },
                            StreamUtils::selectRight);

                    builder.append(integer)
                            .append(",")
                            .append(state.mse())
                            .append(",")
                            .append(network1.toCSV())
                            .append("\n");
                    return state.network();
                }, StreamUtils::selectRight);

        Files.writeString(Path.of(".", "temp.csv"), builder);

        var after = measure(trainingData, trained);
        System.out.println(before + "% " + after + "%");
    }

    private static double measure(Data<Integer> trainingData, Network trained) {
        var totalCorrect = trainingData.stream().mapToInt(entry -> {
            var outputVector = trained.forward(trainingData, entry.getKey());
            var actual = outputVector.apply(0) * 1000d;
            var expected = entry.getValue();

            if (Math.abs(((int) actual) - expected) < 10) {
                return 1;
            } else {
                return 0;
            }
        }).sum();

        var percentage = (double) totalCorrect / trainingData.size();
        return percentage * 100;
    }

    private static GraphNetwork build() {
        var builder = new GraphNetworkBuilder();
        builder.createLayer(1, 1);
        return builder.toNetwork();
    }

    static Vector computeExpected(int value) {
        return Vector.from(value);
    }
}
