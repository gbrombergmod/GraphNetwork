package com.meti;

public class NetMath {
    static double sigmoidDerivative(double hiddenValue) {
        return hiddenValue * (1 - hiddenValue);
    }

    static double sigmoid(double value) {
        return 1d / (1d + Math.pow(Math.E, -value));
    }
}
