package com.meti;

public interface Network {
    Network zero();

    Network addEven(Node node);

    Network addHidden(Node node);

    default Network divide(double scalar) {
        return multiply(1d / scalar);
    }

    Network multiply(double scalar);

    Network subtract(Network other);

    Network addOdd(Node node);

    Network addHidden1(Node node);

    Network addHidden2(Node node);

    Node getHidden();

    Node getHidden1();

    Node getHidden2();

    Node getEven();

    Node getOdd();

    Node apply(int key);
}
