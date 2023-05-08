package com.meti;

public interface Gradients {
    double locateBase(Integer destination);

    Nodes toNodes();

    Gradients add(int id, double baseGradient, Node gradient);
}
