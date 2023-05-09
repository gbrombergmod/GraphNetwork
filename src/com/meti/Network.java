package com.meti;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Stream;

public interface Network {

    Network addRandomConnection();

    <T> NetworkState trainBatch(Data<T> trainingData, List<Map.Entry<Integer, T>> batch, Function<T, Vector> expected);

    <T> Vector forward(Data<T> data, int rawInput);

    <T> NetworkState train(Data<T> data, int key, T value, Function<T, Vector> expected, NetworkState gradientSum);

    Gradients backward(Vector inputVector, List<Integer> topology, Calculations results, double costDerivative);

    Gradients backwards(Gradients gradients, int id, Vector inputVector, Calculations results, double costDerivative);

    Calculations forward(Vector inputVector, List<Integer> topology);

    Calculations forward(Vector inputVector, Calculations calculations1, Integer id);

    double findWeight(int source, int destination);

    boolean isRoot(Integer id);

    List<Integer> listSources(int id);

    Map.Entry<Integer, Integer> findRandomConnection();

    Network zero();

    default Network divide(double scalar) {
        return multiply(1d / scalar);
    }

    Network add(Nodes other);

    Network multiply(double scalar);

    Network subtract(Network other);

    Node apply(int key);

    List<List<Integer>> computeByDepthsForward();

    String toCSV();

    Stream<Integer> stream();

    Network removeConnection(int source, int destination);

    int add(Node node);

    Network addConnection(int source, int destination);

    Map<Integer, List<Integer>> topology();

    int findRandomHiddenNode();

    List<Integer> findParents(int id);

    List<Integer> findChildren(int id);
}
