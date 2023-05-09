package com.meti;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

public record MapNodes(Map<Integer, Node> nodes) implements Nodes {
    private MapNodes() {
        this(new HashMap<>());
    }

    public static Nodes empty() {
        return new MapNodes();
    }

    @Override
    public String toString() {
        return nodes.values()
                .stream()
                .map(Objects::toString)
                .collect(Collectors.joining(","));
    }

    @Override
    public MapNodes add(Nodes other) {
        return new MapNodes(nodes.entrySet()
                .stream().collect(Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> entry.getValue().add(other.apply(entry.getKey())))));
    }

    @Override
    public Nodes addToNode(int id, Node other) {
        var copy = new HashMap<>(nodes());
        if(copy.containsKey(id)) {
            copy.computeIfPresent(id, (integer, node) -> node.add(other));
        } else {
            copy.put(0, other);
        }

        return new MapNodes(copy);
    }

    @Override
    public Node apply(int id) {
        return nodes.get(id);
    }
}