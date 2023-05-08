package com.meti;

public record Gradients(Calculations baseGradients, Nodes nodeGradients) {
    public Gradients() {
        this(MapCalculations.empty(), new MapNodes());
    }

    public Gradients add(int id, double baseGradient, Node gradient) {
        var newBaseGradients = baseGradients.insert(id, baseGradient);
        var newNodeGradients = nodeGradients.addToNode(id, gradient);
        return new Gradients(newBaseGradients, newNodeGradients);
    }
}
