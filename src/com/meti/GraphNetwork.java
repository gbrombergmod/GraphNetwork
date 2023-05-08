package com.meti;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GraphNetwork implements Network {
    private final Map<Integer, List<Integer>> topology;
    private final Nodes nodes;

    public GraphNetwork(Nodes nodes, Map<Integer, List<Integer>> topology) {
        this.nodes = nodes;
        this.topology = topology;
    }

    @Override
    public Network zero() {
        return computeIfPresent(entry -> entry.getValue().zero());
    }

    @Override
    public Network addToNode(int id, Node gradient) {
        return new GraphNetwork(nodes.addToNode(id, gradient), topology);
    }

    @Override
    public Network add(Nodes other) {
        return new GraphNetwork(nodes.add(other), topology);
    }

    @Override
    public Network multiply(double scalar) {
        return computeIfPresent(entry -> entry.getValue().multiply(scalar));
    }

    private GraphNetwork computeIfPresent(Function<Map.Entry<Integer, Node>, Node> mapper) {
        return new GraphNetwork(new Nodes(nodes.nodes().entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, mapper))), topology);
    }

    @Override
    public Network subtract(Network other) {
        return computeIfPresent(entry -> entry.getValue().subtract(other.apply(entry.getKey())));
    }

    @Override
    public Node apply(int key) {
        return nodes.nodes().get(key);
    }

    @Override
    public Stream<Integer> streamConnections(int id) {
        return topology.get(id).stream();
    }

    public List<List<Node>> sortByDepthForwards() {
        var depthMap = computeDepthMap();
        var list = new ArrayList<>(depthMap.keySet());
        Collections.sort(list);
        return list.stream().map(depthMap::get).collect(Collectors.toList());
    }

    private TreeMap<Integer, List<Node>> computeDepthMap() {
        Map<Integer, Integer> depthMap = new HashMap<>();
        Map<Integer, Boolean> visited = new HashMap<>();
        Queue<Integer> queue = new LinkedList<>();

        // Initialize visited map
        for (Integer key : nodes.nodes().keySet()) {
            visited.put(key, false);
        }

        // Find all root nodes (nodes without incoming edges)
        Set<Integer> allNodes = nodes.nodes().keySet();
        Set<Integer> nonRootNodes = topology.values().stream().flatMap(List::stream).collect(Collectors.toSet());
        Set<Integer> rootNodes = new HashSet<>(allNodes);
        rootNodes.removeAll(nonRootNodes);

        // BFS from each root node to calculate depth of each node
        for (Integer root : rootNodes) {
            queue.add(root);
            visited.put(root, true);
            depthMap.put(root, 0);
            while (!queue.isEmpty()) {
                Integer id = queue.poll();
                List<Integer> neighbours = topology.get(id);
                if (neighbours != null) {
                    for (Integer neighbour : neighbours) {
                        if (!visited.get(neighbour)) {
                            queue.add(neighbour);
                            visited.put(neighbour, true);
                            depthMap.put(neighbour, depthMap.get(id) + 1);
                        }
                    }
                }
            }
        }

        // Group nodes by depth
        return depthMap.entrySet().stream()
                .collect(Collectors.groupingBy(
                        Map.Entry::getValue,
                        TreeMap::new,
                        Collectors.mapping(entry -> nodes.nodes().get(entry.getKey()), Collectors.toList())
                ));
    }

    public List<List<Node>> sortBackwards() {
        var copy = sortByDepthForwards();
        Collections.reverse(copy);
        return copy;
    }
}
