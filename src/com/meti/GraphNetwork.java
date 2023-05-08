package com.meti;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GraphNetwork implements Network {
    public static final int EVEN_ID = 3;
    public static final int HIDDEN_ID = 0;
    public static final int ODD_ID = 4;
    public static final int HIDDEN1_ID = 1;
    public static final int HIDDEN2_ID = 2;
    private final Map<Integer, Node> nodes;

    public GraphNetwork(Map<Integer, Node> nodes) {
        this.nodes = nodes;
    }

    @Override
    public Network zero() {
        return computeIfPresent(entry -> entry.getValue().zero());
    }

    @Override
    public GraphNetwork addToNode(Node gradient, int id) {
        var copy = new HashMap<>(nodes);
        copy.computeIfPresent(id, (key, node) -> node.add(gradient));
        return new GraphNetwork(copy);
    }

    @Override
    public Network multiply(double scalar) {
        return computeIfPresent(entry -> entry.getValue().multiply(scalar));
    }

    private GraphNetwork computeIfPresent(Function<Map.Entry<Integer, Node>, Node> mapper) {
        return new GraphNetwork(nodes.entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, mapper)));
    }

    @Override
    public Network subtract(Network other) {
        return computeIfPresent(entry -> entry.getValue().subtract(other.apply(entry.getKey())));
    }

    @Override
    public Node apply(int key) {
        return nodes.get(key);
    }
}
