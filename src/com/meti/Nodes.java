package com.meti;

import java.util.Map;

public interface Nodes {
    Nodes add(Nodes other);

    Nodes addToNode(int id, Node other);

    Node apply(int id);
}
