package com.meti;

import java.util.List;

public interface Calculations {
    Calculations insert(int id, double result);

    Vector locate(List<Integer> ids);

    double locate(int id);
}
