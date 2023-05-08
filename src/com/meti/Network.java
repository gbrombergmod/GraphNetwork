package com.meti;

import java.util.Objects;

final class Network {
    private final Node hidden;
    private final Node hidden1;
    private final Node hidden2;
    private final Node even;
    private final Node odd;

    Network(Node hidden, Node hidden1, Node hidden2, Node even, Node odd) {
        this.hidden = hidden;
        this.hidden1 = hidden1;
        this.hidden2 = hidden2;
        this.even = even;
        this.odd = odd;
    }

    public static Network random() {
        return new Network(Node.random(1), Node.random(1), Node.random(1),
                Node.random(3), Node.random(3));
    }

    static Network zero() {
        return new Network(
                Node.zero(1), Node.zero(1), Node.zero(1),
                Node.zero(3), Node.zero(3));
    }

    public Network addEven(Node outputGradient) {
        return new Network(getHidden(), getHidden1(), getHidden2(), getEven().add(outputGradient), getOdd());
    }

    public Network addHidden(Node node) {
        return new Network(getHidden().add(node), getHidden1(), getHidden2(), getEven(), getOdd());
    }

    public Network divide(double scalar) {
        return new Network(getHidden().divide(scalar), getHidden1().divide(scalar), getHidden2().divide(scalar),
                getEven().divide(scalar), getOdd().divide(scalar));
    }

    public Network multiply(double scalar) {
        return new Network(getHidden().multiply(scalar), getHidden1().multiply(scalar), getHidden1().multiply(scalar),
                getEven().divide(scalar), getOdd().divide(scalar));
    }

    public Network subtract(Network other) {
        return new Network(getHidden().subtract(other.getHidden()), getHidden1().subtract(other.getHidden1()), getHidden2().subtract(other.getHidden2()),
                getEven().subtract(other.getEven()), getOdd().subtract(other.getOdd()));
    }

    public Network addOdd(Node odd) {
        return new Network(getHidden(), getHidden1(), getHidden2(), getEven(), odd);
    }

    public Network addHidden1(Node hidden1) {
        return new Network(getHidden(), hidden1.add(hidden1), getHidden2(), getEven(), getOdd());
    }

    public Network addHidden2(Node hidden2) {
        return new Network(getHidden(), getHidden1(), hidden2.add(hidden2), getEven(), getOdd());
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (Network) obj;
        return Objects.equals(this.getHidden(), that.getHidden()) &&
               Objects.equals(this.getHidden1(), that.getHidden1()) &&
               Objects.equals(this.getHidden2(), that.getHidden2()) &&
               Objects.equals(this.getEven(), that.getEven()) &&
               Objects.equals(this.getOdd(), that.getOdd());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getHidden(), getHidden1(), getHidden2(), getEven(), getOdd());
    }

    @Override
    public String toString() {
        return "Network[" +
               "hidden=" + getHidden() + ", " +
               "hidden1=" + getHidden1() + ", " +
               "hidden2=" + getHidden2() + ", " +
               "even=" + getEven() + ", " +
               "odd=" + getOdd() + ']';
    }

    public Node getHidden() {
        return hidden;
    }

    public Node getHidden1() {
        return hidden1;
    }

    public Node getHidden2() {
        return hidden2;
    }

    public Node getEven() {
        return even;
    }

    public Node getOdd() {
        return odd;
    }
}
