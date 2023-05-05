import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {
    public static void main(String[] args) {
        var data = IntStream.range(0, 1000)
                .boxed()
                .collect(Collectors.toMap(Function.identity(), value -> value % 2 == 0));

        var weight = Math.random();
        var bias = Math.random();

        var totalCorrect = data.entrySet().stream().mapToInt(entry -> {
            var input = (double) entry.getKey() / (double) data.size();
            var evaluated = weight * input + bias;
            var activated = 1d / (1d + Math.pow(Math.E, -evaluated));

            var isEven = entry.getValue();
            if (isEven && activated < 0.5) {
                return 1;
            } else if (!isEven && activated >= 0.5) {
                return 1;
            } else {
                return 0;
            }
        }).sum();

        System.out.println(totalCorrect);
    }
}