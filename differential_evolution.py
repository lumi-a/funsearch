import random
from specs.nemhauser_ullmann import evaluate_instance

bounds = []
for _ in range(17):
    bounds.extend([(1, 2**16), (1, 2**16)])


def to_maximise(xs: list[float]):
    items = [(int(xs[i]), int(xs[i + 1])) for i in range(0, len(xs), 2)]
    return -evaluate_instance(items)


def differential_evolution(to_maximise, bounds):
    dimension = len(bounds)
    population_size = 10 * dimension
    crossover_probability = 0.9
    differential_weight = 0.8

    population = [[a + random.random() * (b - a) for (a, b) in bounds] for _ in range(population_size)]
    scores = [to_maximise(x) for x in population]

    iteration = 0
    while True:
        for x_index, x in enumerate(population):
            a, b, c = random.sample(population, 3)
            random_index = random.randint(1, dimension)
            y = [
                a[i] + differential_weight * (b[i] - c[i])
                if (random.random() < crossover_probability or i == random_index)
                else x[i]
                for i in range(dimension)
            ]
            for i, (a, b) in enumerate(bounds):
                y[i] = max(a, min(b, y[i]))

            new_score = to_maximise(y)
            if new_score <= scores[x_index]:
                population[x_index] = y
                scores[x_index] = new_score

        iteration += 1
        if iteration % 10 == 0:
            best_ix = max(range(len(scores)), key=scores.__getitem__)
            best = population[best_ix]
            print(iteration, to_maximise(best), sep="\t")
            print(best)


if __name__ == "__main__":
    differential_evolution(to_maximise, bounds)
