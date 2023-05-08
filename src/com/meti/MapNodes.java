package com.meti;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public record MapNodes(Map<Integer, Node> nodes) implements Nodes {
    public MapNodes() {
        this(new HashMap<>());
    }

    @Override
    public MapNodes add(Nodes other) {
        return new MapNodes(nodes.entrySet()
                .stream().collect(Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> entry.getValue().add(nodes.get(entry.getKey())))));
    }

    @Override
    public MapNodes addToNode(int id, Node other) {
        var copy = new HashMap<>(nodes());
        if(copy.containsKey(id)) {
            copy.computeIfPresent(id, (integer, node) -> node.add(other));
        } else {
            copy.put(0, other);
        }

        return new MapNodes(copy);
    }
}