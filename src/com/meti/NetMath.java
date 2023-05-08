package com.meti;

public class NetMath {
    static double sigmoidDerivative(double hiddenValue) {
        return hiddenValue * (1 - hiddenValue);
    }

    static double sigmoid(double value) {
        return 1d / (1d + Math.pow(Math.E, -value));
    }

    static double reluDerivative(double value) {
        if (value > 0) return 1;
        else return 0;
    }

    static double relu(double value) {
        if (value > 0) return value;
        else return 0;
    }

    static double randomFloat() {
        return Math.random();
    }
}
