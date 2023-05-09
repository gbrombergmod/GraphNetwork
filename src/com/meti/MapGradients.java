package com.meti;

public record MapGradients(Calculations baseGradients, Nodes nodeGradients) implements Gradients {
    private MapGradients() {
        this(MapCalculations.empty(), MapNodes.empty());
    }

    public static Gradients empty() {
        return new MapGradients();
    }

    @Override
    public double locateBase(Integer destination) {
        return baseGradients.locate(destination);
    }

    @Override
    public Nodes asNodes() {
        return nodeGradients;
    }

    @Override
    public Gradients add(int id, double baseGradient, Node gradient) {
        var newBaseGradients = baseGradients.insert(id, baseGradient);
        var newNodeGradients = nodeGradients.addToNode(id, gradient);
        return new MapGradients(newBaseGradients, newNodeGradients);
    }

    @Override
    public Gradients backwardsOutput(int id, Vector inputs, double output, double upstreamDerivative) {
        var activated = NetMath.sigmoidDerivative(output);
        var base = upstreamDerivative * activated;
        var gradient = new Node(inputs, 1d).multiply(base);
        return add(id, upstreamDerivative * activated, gradient);
    }
}
