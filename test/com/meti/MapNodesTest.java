package com.meti;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class MapNodesTest {

    @Test
    void add() {
        var first = Node.random(0);
        var second = Node.random(0);
        var expected = first.add(second);

        var firstNodes = new MapNodes(Map.of(0, first));
        var secondNodes = new MapNodes(Map.of(0, second));

        var actual = firstNodes.add(secondNodes).apply(0);
        assertEquals(expected, actual);
    }
}