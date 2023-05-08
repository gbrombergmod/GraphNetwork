package com.meti;

public interface Network {
    Network zero();

    Network addEven(Node outputGradient);

    Network addHidden(Node node);

    Network divide(double scalar);

    Network multiply(double scalar);

    Network subtract(Network other);

    Network addOdd(Node odd);

    Network addHidden1(Node hidden1);

    Network addHidden2(Node hidden2);

    Node getHidden();

    Node getHidden1();

    Node getHidden2();

    Node getEven();

    Node getOdd();
}
