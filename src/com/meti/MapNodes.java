package com.meti;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public record MapNodes(Map<Integer, Node> nodes) implements Nodes {
    private MapNodes() {
        this(new HashMap<>());
    }

    public static Nodes empty() {
        return new MapNodes();
    }

    @Override
    public MapNodes add(Nodes other) {
        if(!nodes.keySet().containsAll(other.ids()) || !other.ids().containsAll(nodes.keySet())) {
            var format = "Argument mismatch: '%s', '%s'.";
            var message = format.formatted(this, other);
            throw new IllegalArgumentException(message);
        }

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
        if(!nodes.containsKey(id)) {
            var format = "Node of id '%d' is not found in the graph: %s";
            var message = format.formatted(id, nodes);
            throw new IllegalArgumentException(message);
        }
        return nodes.get(id);
    }

    @Override
    public Collection<Integer> ids() {
        return nodes.keySet();
    }

    MapNodes computeIfPresent(Function<Map.Entry<Integer, Node>, Node> mapper) {
        return new MapNodes(nodes().entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, mapper)));
    }
}