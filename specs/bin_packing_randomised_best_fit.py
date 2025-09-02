"""I'm trying to find instances of the bin-packing problem where, if the input is shuffled, the best-fit online-heuristic performs poorly in expectation. All bins have capacity 1.0.

To generate instances that best-fit performs poorly on, I have tried the following functions so far. Please write another one that returns an instance and is similar, but has some lines altered.
"""

import math
import funsearch


@funsearch.run
def evaluate(_: int) -> float:
    """Returns the estimated score of the instance."""
    instance = get_items()
    # Assert determinancy
    if instance != get_items() or len(instance) == 0:
        return 0.0
    instance = [min(1.0, item) for item in instance if item > 0]
    return evaluate_instance(instance)


def evaluate_instance(instance: list[float]) -> float:
    """Returns the estimated score of the instance."""
    import random
    import re
    import subprocess
    from pathlib import Path
    import threading

    ident = threading.get_ident()

    def optimum_value(instance: list[float]) -> int:
        scaled = [math.floor(item * 1_000_000) for item in instance]

        # This requires installing https://github.com/fontanf/packingsolver
        Path(f"items-{ident}.csv").write_text("X\n" + "\n".join(map(str, scaled)))
        Path(f"bins-{ident}.csv").write_text(f"X,COPIES\n1000000,{len(instance)}")
        Path(f"parameters-{ident}.csv").write_text("NAME,VALUE\nobjective,bin-packing")
        stdout = subprocess.check_output(
            [
                r"C:\a\packingsolver\install\bin\packingsolver_onedimensional.exe",
                "--items",
                f"items-{ident}.csv",
                "--bins",
                f"bins-{ident}.csv",
                "--parameters",
                f"parameters-{ident}.csv",
                "--time-limit",
                "1",
            ],
        ).decode()
        # Remove files
        Path(f"items-{ident}.csv").unlink()
        Path(f"bins-{ident}.csv").unlink()
        Path(f"parameters-{ident}.csv").unlink()
        bins = int(re.search(r"Number of bins:\s*(\d+) / \d+ .*", stdout).group(1))
        if bins == 0:
            print(stdout)
        return bins

    def best_fit(instance: list[float]) -> int:
        bins = []  # There are more efficient datastructures.
        for item in instance:
            best_bin_ix = None
            best_bin_size = -1
            for i, old_bin in enumerate(bins):
                new_bin = old_bin + item
                # Avoid floating-point rounding issues
                if new_bin <= 1 + 1e-9 and new_bin > best_bin_size:
                    best_bin_ix = i
                    best_bin_size = new_bin

            if best_bin_ix is not None:
                bins[best_bin_ix] += item
            else:
                bins.append(item)

        return len(bins)

    total = 0
    samples = 10_000
    for _ in range(samples):
        random.shuffle(instance)
        total += best_fit(instance)
    mean = total / samples
    return mean / optimum_value(instance)


@funsearch.evolve
def get_items() -> list[float]:
    """Return a new bin-packing-instance, specified by the list of items.

    The items must be floats between 0 and 1.
    """
    items = [0.4, 0.5, 0.6]
    return items
