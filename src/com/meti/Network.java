package com.meti;

import java.util.List;
import java.util.stream.Stream;

public interface Network {
    Stream<Integer> streamConnections(int id);

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
