package com.meti;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GraphNetwork implements Network {
    private final Map<Integer, List<Integer>> topology;
    private final MapNodes nodes;

    public GraphNetwork(MapNodes nodes, Map<Integer, List<Integer>> topology) {
        this.nodes = nodes;
        this.topology = topology;
    }

    @Override
    public Map.Entry<Integer, Integer> randomConnection() {
        if (topology.isEmpty()) {
            throw new IllegalStateException("Empty topology.");
        }

        var sources = topology.keySet().stream().toList();
        var randomSource = sources.get((int) (Math.random() * sources.size()));
        var randomDestinations = topology.get(randomSource);
        var randomDestination = randomDestinations.get((int) (Math.random() * randomDestinations.size()));
        return new AbstractMap.SimpleEntry<>(randomSource, randomDestination);
    }

    @Override
    public Network zero() {
        return computeIfPresent(entry -> entry.getValue().zero());
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
        return new GraphNetwork(nodes.computeIfPresent(mapper), topology);
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
    public List<Integer> listSources(int id) {
        return topology.entrySet().stream()
                .flatMap(entry -> entry.getValue().contains(id) ? Stream.of(entry.getKey()) : Stream.empty())
                .collect(Collectors.toList());
    }

    @Override
    public List<List<Integer>> computeByDepthsForward() {
        var depthMap = computeDepthMap();
        var list = new ArrayList<>(depthMap.keySet());
        Collections.sort(list);
        return list.stream()
                .map(depthMap::get)
                .collect(Collectors.toList());
    }

    @Override
    public String toCSV() {
        return nodes.toCSV();
    }

    @Override
    public Stream<Integer> stream() {
        return nodes.ids().stream().sorted();
    }

    @Override
    public Network removeConnection(int source, int destination) {
        var destinations = topology.get(source);
        var destinationsCopy = new ArrayList<>(destinations);
        if (destinationsCopy.contains(destination)) {
            destinationsCopy.remove((Object) destination);
        }

        var topologyCopy = new HashMap<>(topology);
        topologyCopy.put(source, destinationsCopy);
        return new GraphNetwork(nodes, topologyCopy);
    }

    @Override
    public int add(Node node) {
        return nodes.add(node);
    }

    @Override
    public Network addConnection(int source, int destination) {
        if (!topology.containsKey(source)) {
            topology.put(source, new ArrayList<>());
        }

        topology.get(source).add(destination);
        return this;
    }

    @Override
    public Map<Integer, List<Integer>> topology() {
        return topology;
    }

    private Map<Integer, List<Integer>> computeDepthMap() {
        Map<Integer, Integer> depthMap = new HashMap<>();
        Map<Integer, Boolean> visited = new HashMap<>();
        Queue<Integer> queue = new LinkedList<>();

        // Initialize visited map
        for (Integer key : nodes.nodes().keySet()) {
            visited.put(key, false);
        }

        // Find all root nodeGradients (nodeGradients without incoming edges)
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

        // Group nodeGradients by depth
        var copy = new HashMap<Integer, List<Integer>>();
        for (var entry : depthMap.entrySet()) {
            var depth = entry.getValue();
            var nodeID = entry.getKey();
            if (copy.containsKey(depth)) {
                copy.get(depth).add(nodeID);
            } else {
                var list = new ArrayList<Integer>();
                list.add(nodeID);
                copy.put(depth, list);
            }
        }

        return copy;
    }

    @Override
    public String toString() {
        return nodes.toString();
    }

    @Override
    public <T> NetworkState trainBatch(Data<T> trainingData, List<Map.Entry<Integer, T>> batch, Function<T, Vector> expected) {
        var state = batch.stream()
                .reduce(new NetworkState(zero(), 0d),
                        (gradientSum, entry) -> train(trainingData, entry.getKey(), entry.getValue(), expected, gradientSum),
                        StreamUtils::selectRight);

        var batchSize = batch.size();
        var gradient = state.network()
                .divide(batchSize)
                .multiply(Main.LEARNING_RATE);

        return new NetworkState(subtract(gradient), state.mse() / batchSize);
    }

    @Override
    public Vector forward(Data data, int rawInput) {
        var layers = computeByDepthsForward()
                .stream()
                .toList();

        var topology = layers
                .stream()
                .flatMap(Collection::stream)
                .toList();

        var input = data.normalize(rawInput);
        var inputVector = Vector.from(input);
        return forward(inputVector, topology).locate(layers.get(layers.size() - 1));
    }

    @Override
    public <T> NetworkState train(Data<T> data, int key, T value, Function<T, Vector> expected, NetworkState gradientSum) {
        var lists = computeByDepthsForward();
        var topology = lists
                .stream()
                .flatMap(Collection::stream)
                .toList();

        var input = data.normalize(key);
        var inputVector = Vector.from(input);
        var results = forward(inputVector, topology);

        var actual = results.locate(lists.get(lists.size() - 1));
        var divide = expected.apply(value).divide(1000d);

        var error = actual.subtract(divide).sum();
        var mse = Math.pow(error, 2d);

        var costDerivative = 2 * error;

        var gradients = backward(inputVector, topology, results, costDerivative);
        var gradientsAsNodes = gradients.asNodes();
        return new NetworkState(gradientSum.network().add(gradientsAsNodes), mse);
    }

    @Override
    public Gradients backward(Vector inputVector, List<Integer> topology, Calculations results, double costDerivative) {
        var copy = new ArrayList<>(topology);
        Collections.reverse(copy);
        return copy.stream().reduce(MapGradients.empty(),
                (gradients, integer) -> backwards(gradients, integer, inputVector, results, costDerivative),
                (previous, next) -> next);
    }

    @Override
    public Gradients backwards(Gradients gradients, int id, Vector inputVector, Calculations results, double costDerivative) {
        if (isRoot(id)) {
            var sources = listSources(id);
            double previousDerivative;
            if (sources.isEmpty()) {
                previousDerivative = costDerivative;
            } else {
                previousDerivative = sources
                        .stream()
                        .mapToDouble(destination -> gradients.locateBase(destination) * findWeight(id, destination))
                        .sum();
            }

            return gradients.backwardsOutput(id, inputVector, results.locate(id), previousDerivative);
        } else {
            var ids = listSources(id);
            var previousInputs = results.locate(ids);
            return gradients.backwardsOutput(id, previousInputs, results.locate(id), costDerivative);
        }
    }

    @Override
    public Calculations forward(Vector inputVector, List<Integer> topology) {
        return topology.stream().reduce(MapCalculations.empty(),
                (calculations1, id) -> forward(inputVector, calculations1, id),
                (previous, next) -> next);
    }

    @Override
    public Calculations forward(Vector inputVector, Calculations calculations1, Integer id) {
        Vector layer;
        if (isRoot(id)) {
            layer = inputVector;
        } else {
            var collect = listSources(id);
            layer = calculations1.locate(collect);
        }

        var forward = apply(id).forward(layer);
        return calculations1.insert(id, forward);
    }

    @Override
    public double findWeight(int source, int destination) {
        var integers = listSources(source);
        var index = integers.indexOf(destination);
        return apply(destination).weight().apply(index);
    }

    @Override
    public boolean isRoot(Integer id) {
        return listSources(id)
                .stream()
                .findAny()
                .isEmpty();
    }
}
