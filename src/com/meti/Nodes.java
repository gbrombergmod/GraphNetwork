package com.meti;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

public record Nodes(Map<Integer, Node> nodes) {
    public Nodes() {
        this(new HashMap<>());
    }

    public Nodes add(Nodes other) {
        return new Nodes(nodes.entrySet()
                .stream().collect(Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> entry.getValue().add(nodes.get(entry.getKey())))));
    }

    Nodes addToNode(int id, Node other) {
        var copy = new HashMap<>(nodes());
        if(copy.containsKey(id)) {
            copy.computeIfPresent(id, (integer, node) -> node.add(other));
        } else {
            copy.put(0, other);
        }

        return new Nodes(copy);
    }
}