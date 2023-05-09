package com.meti;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MapGradientsTest {

    @Test
    void backwardsOutput() {
        var gradients = MapGradients.empty()
                .backwardsOutput(0, Vector.from(5), 10, 2);

        var baseGradient = -180;
        var node = new Node(Vector.from(-900), baseGradient);
        var actual = MapGradients.empty().add(0, baseGradient, node);
        assertEquals(gradients, actual);
    }
}