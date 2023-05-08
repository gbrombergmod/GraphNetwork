package com.meti;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

public interface Network {

    NetworkState trainBatch(Data trainingData, List<Map.Entry<Integer, Boolean>> batch, Function<Boolean, Vector> expected);

    Vector forward(Data data, int rawInput);

    NetworkState train(Data data, int key, boolean value, Function<Boolean, Vector> mapper);

    Gradients backward(Vector inputVector, List<Integer> topology, Calculations results, double costDerivative);

    Gradients backwards(Gradients gradients, int id, Vector inputVector, Calculations results, double costDerivative);

    Calculations forward(Vector inputVector, List<Integer> topology);

    Calculations forward(Vector inputVector, Calculations calculations1, Integer id);

    double findWeight(int source, int destination);

    boolean isRoot(Integer id);

    List<Integer> listSources(int id);

    Network zero();

    default Network divide(double scalar) {
        return multiply(1d / scalar);
    }

    Network add(Nodes other);

    Network multiply(double scalar);

    Network subtract(Network other);

    Node apply(int key);

    List<List<Integer>> computeByDepthsForward();
}
