package com.meti;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NetMathTest {

    @Test
    void sigmoidDerivative() {
        var expected = -90d;
        var actual = NetMath.sigmoidDerivative(10);
        assertEquals(expected, actual);
    }
}