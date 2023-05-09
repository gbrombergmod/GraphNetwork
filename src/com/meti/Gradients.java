package com.meti;

public interface Gradients {
    Gradients backwardsOutput(int id, Vector inputs, double output, double upstreamDerivative);

    double locateBase(Integer destination);

    Nodes toNodes();

    Gradients add(int id, double baseGradient, Node gradient);
}
