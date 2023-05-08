package com.meti;

import java.util.stream.Stream;

public interface Network {
    Stream<Integer> streamConnections(int id);

    Network zero();

    GraphNetwork addToNode(Node gradient, int id);

    default Network divide(double scalar) {
        return multiply(1d / scalar);
    }

    Network multiply(double scalar);

    Network subtract(Network other);

    Node apply(int key);
}
