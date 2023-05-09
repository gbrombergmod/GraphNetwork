package com.meti;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GraphNetworkTest {
    @Test
    void backward() {

    }

    @Test
    void testSortByDepthForwards_singleRoot() {
        var nodes = new MapNodes(Map.of(
                1, new Node(1),
                2, new Node(2),
                3, new Node(3),
                4, new Node(4)
        ));

        var topology = Map.of(
                1, List.of(2),
                2, List.of(3),
                3, List.of(4)
        );

        var graphNetwork = new GraphNetwork(nodes, topology);
        var sortedNodes = graphNetwork.computeByDepthsForward();

        assertEquals(4, sortedNodes.size());
        assertEquals(List.of(1), sortedNodes.get(0));
        assertEquals(List.of(2), sortedNodes.get(1));
        assertEquals(List.of(3), sortedNodes.get(2));
        assertEquals(List.of(4), sortedNodes.get(3));
    }

    @Test
    void testSortByDepthForwards_multipleRoots() {
        var nodes = new MapNodes(Map.of(
                1, new Node(1),
                2, new Node(2),
                3, new Node(3),
                4, new Node(4),
                5, new Node(5),
                6, new Node(6)
        ));

        var topology = Map.of(
                1, List.of(2),
                2, List.of(3),
                4, List.of(5),
                5, List.of(6)
        );

        var graphNetwork = new GraphNetwork(nodes, topology);
        var sortedNodes = graphNetwork.computeByDepthsForward();

        assertEquals(3, sortedNodes.size());
        assertTrue(sortedNodes.get(0).containsAll(List.of(1, 4)));
        assertTrue(sortedNodes.get(1).containsAll(List.of(2, 5)));
        assertTrue(sortedNodes.get(2).containsAll(List.of(3, 6)));
    }
}