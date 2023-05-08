package com.meti;

import java.util.List;

public interface Network {
    default double findWeight(int source, int destination) {
        var integers = listConnections(source);
        var index = integers.indexOf(destination);
        return apply(destination).weight().apply(index);
    }

    default boolean isRoot(Integer id) {
        return listConnections(id)
                .stream()
                .toList().isEmpty();
    }

    List<Integer> listConnections(int id);

    Network zero();

    Network addToNode(int id, Node gradient);

    default Network divide(double scalar) {
        return multiply(1d / scalar);
    }

    Network add(Nodes other);

    Network multiply(double scalar);

    Network subtract(Network other);

    Node apply(int key);

    List<List<Integer>> computeByDepthsForward();
}
