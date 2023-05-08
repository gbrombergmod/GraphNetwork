package com.meti;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GraphNetwork implements Network {
    private final Map<Integer, List<Integer>> topology;
    private final Nodes nodes;

    public GraphNetwork(Nodes nodes, Map<Integer, List<Integer>> topology) {
        this.nodes = nodes;
        this.topology = topology;
    }

    @Override
    public Network zero() {
        return computeIfPresent(entry -> entry.getValue().zero());
    }

    @Override
    public Network addToNode(int id, Node gradient) {
        return new GraphNetwork(nodes.addToNode(id, gradient), topology);
    }

    @Override
    public Network add(Nodes other) {
        return new GraphNetwork(nodes.add(other), topology);
    }

    @Override
    public Network multiply(double scalar) {
        return computeIfPresent(entry -> entry.getValue().multiply(scalar));
    }

    private GraphNetwork computeIfPresent(Function<Map.Entry<Integer, Node>, Node> mapper) {
        return new GraphNetwork(new Nodes(nodes.nodes().entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, mapper))), topology);
    }

    @Override
    public Network subtract(Network other) {
        return computeIfPresent(entry -> entry.getValue().subtract(other.apply(entry.getKey())));
    }

    @Override
    public Node apply(int key) {
        return nodes.nodes().get(key);
    }

    @Override
    public Stream<Integer> streamConnections(int id) {
        return topology.get(id).stream();
    }
}
