package com.meti;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GraphNetwork implements Network {
    private final Map<Integer, Node> nodes;
    private final Map<Integer, List<Integer>> topology;

    public GraphNetwork(Map<Integer, Node> nodes, Map<Integer, List<Integer>> topology) {
        this.nodes = nodes;
        this.topology = topology;
    }

    @Override
    public Network zero() {
        return computeIfPresent(entry -> entry.getValue().zero());
    }

    @Override
    public GraphNetwork addToNode(Node gradient, int id) {
        var copy = new HashMap<>(nodes);
        copy.computeIfPresent(id, (key, node) -> node.add(gradient));
        return new GraphNetwork(copy, topology);
    }

    @Override
    public Network multiply(double scalar) {
        return computeIfPresent(entry -> entry.getValue().multiply(scalar));
    }

    private GraphNetwork computeIfPresent(Function<Map.Entry<Integer, Node>, Node> mapper) {
        return new GraphNetwork(nodes.entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, mapper)), topology);
    }

    @Override
    public Network subtract(Network other) {
        return computeIfPresent(entry -> entry.getValue().subtract(other.apply(entry.getKey())));
    }

    @Override
    public Node apply(int key) {
        return nodes.get(key);
    }

    @Override
    public Stream<Integer> streamConnections(int id) {
        return topology.get(id).stream();
    }
}
