package com.meti;

import java.util.List;
import java.util.Map;

public interface Network {

    Network trainBatch(Data trainingData, List<Map.Entry<Integer, Boolean>> batch);

    Network train(Data data, int key, boolean value);

    Gradients backward(Vector inputVector, List<Integer> topology, Calculations results, double costDerivative);

    Gradients backwards(Gradients gradients, int id, Vector inputVector, Calculations results, double costDerivative);

    Gradients backwardsHidden(Calculations results, int source, Gradients gradientSum, Vector inputs);

    Calculations forward(Vector inputVector, List<Integer> topology);

    Calculations forward(Vector inputVector, Calculations calculations1, Integer id);

    double findWeight(int source, int destination);

    boolean isRoot(Integer id);

    List<Integer> listConnections(int id);

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
