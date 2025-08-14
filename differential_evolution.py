import random
from specs.nemhauser_ullmann import evaluate

bounds = []
for _ in range(10):
    bounds.append((-20.0, 20.0))
    bounds.extend([(-100, 100) for _ in range(3)])

def to_maximise(xs: list[float]):
    items = [(2 ** max(32, min(-32, xs[i])), xs[(i + 1) : (i + 4)]) for i in range(0, len(xs), 4)]
    return -evaluate(items)


def differential_evolution(to_maximise, bounds):
    dimension = len(bounds)
    population_size = 10 * dimension
    crossover_probability = 0.9
    differential_weight = 0.8

    population = [[a + random.random() * (b - a) for (a, b) in bounds] for _ in range(population_size)]

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
            if to_maximise(y) <= to_maximise(x):
                population[x_index] = y

        iteration += 1
        if iteration % 10 == 0:
            best = max(population, key=to_maximise)
            print(iteration, to_maximise(best), sep="\t")
            print(best)


if __name__ == "__main__":
    differential_evolution(to_maximise, bounds)
