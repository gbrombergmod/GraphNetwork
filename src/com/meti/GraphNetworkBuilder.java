package com.meti;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record GraphNetworkBuilder(Map<Integer, Node> nodes, Map<Integer, List<Integer>> topology) {
    List<Integer> createLayer(int layerSize, int inputSize) {
        return IntStream.range(0, layerSize)
                .mapToObj(value -> {
                    return create(inputSize);
                }).collect(Collectors.toList());
    }

    int create(int inputSize) {
        var id = nodes().size();
        nodes().put(id, Node.random(inputSize));
        return id;
    }

    GraphNetwork toNetwork() {
        return new GraphNetwork(new MapNodes(nodes()), topology());
    }

    public void connect(List<Integer> source, List<Integer> destination) {
        for (var integer : source) {
            topology.put(integer, destination);
        }
    }
}