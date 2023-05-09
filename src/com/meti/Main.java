package com.meti;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
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
        var network = random();
        measure(trainingData, network);

        var builder = new StringBuilder();
        builder.append("EPOCH,MSE,");
        network.stream().forEach(nodeID -> {
            var prefix = "NODE" + nodeID;
            var node = network.apply(nodeID);
            var weightCount = node.weight().size();

            IntStream.range(0, weightCount).forEach(weightIndex -> builder
                    .append(prefix)
                    .append("_WEIGHT")
                    .append(weightIndex)
                    .append(","));

            builder.append(prefix).append("_BIAS,");
        });
        builder.append("\n");

        var trained = IntStream.range(0, EPOCH_COUNT)
                .boxed()
                .reduce(network, (network1, integer) -> {
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

        measure(trainingData, trained);
    }

    private static void measure(Data<Integer> trainingData, Network trained) {
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
        System.out.println((percentage * 100) + "%");
    }

    public static Network random() {
        var nodes = new HashMap<Integer, Node>();
        var topology = new HashMap<Integer, List<Integer>>();

        return getGraphNetwork(new GraphNetworkBuilder(nodes, topology));
    }

    private static GraphNetwork getGraphNetwork(GraphNetworkBuilder builder) {
        var hidden = builder.createLayer(3, 1);
        var hidden1 = builder.createLayer(3, 3);
        var hidden2 = builder.createLayer(3, 3);
        var output = builder.createLayer(1, 3);

        builder.connect(hidden, hidden1);
        builder.connect(hidden1, hidden2);
        builder.connect(hidden2, output);

        return builder.toNetwork();
    }

    static Vector computeExpected(int value) {
        return Vector.from(value);
    }
}
