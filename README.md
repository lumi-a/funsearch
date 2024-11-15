Fork of [jonppe's fork](https://github.com/jonppe/funsearch) of [google-deepmind's funsearch](https://github.com/google-deepmind/funsearch).

## Installation

Requires a [Python](https://www.python.org/) version ≥ 3.9.

```shell
python -m venv .venv        # Create virtual environment
source .venv/bin/activate   # Enter your virtual environment
# On Windows, use `.venv\Scripts\activate`
# Use `deactivate` to exit the environment (any platform)

pip install pdm             # For package- and dependency-management
pdm install --no-self       # Install project's dependencies
pip install -e --no-deps .  # Install project
mkdir data                  # Create directory for storing data
```

## Running

Enter your virtual environment (see above). Consult `llm keys set --help` and set your API-key. The following runs funsearch with OpenAI's `gpt-3.5-turbo`:

```shell
funsearch run --model_name gpt-3.5-turbo --output_path data --sandbox_type ExternalProcessSandbox examples/gasoline_spec.py 11

# Example output:
# INFO:root:Writing logs to data/1731323471
# INFO:absl:Best score of island 0 increased to 1.0
#  ⋮
# INFO:absl:Best score of island 9 increased to 1.0
# INFO:absl:Best score of island 3 increased to 1.2
# INFO:absl:Best score of island 8 increased to 1.0555555555555556
# INFO:absl:Best score of island 4 increased to 1.105263157894737
# INFO:absl:Best score of island 8 increased to 1.2222222222222223
# INFO:root:Keyboard interrupt. Stopping.
# INFO:absl:Saving backup to data/backups/program_db_gasoline_1731323471_0.pickle.
```


Explaining the arguments (run `funsearch run --help` for more options):

- Pick your `--model_name` from the list of available models (`llm models list`). To use other models, see the [llm-docs](https://llm.datasette.io/en/stable/other-models.html).
- The `--output_path` directory stores API-calls and responses.
- The `--sandbox_type` sets the sandbox in which the searched functions will be run. This might be desirable because, for evaluation, we execute python-code that the LLM gave to us without restrictions. The documentation of `/funsearch/sandbox.py` explains the different sandbox types. Above, we used `ExternalProcessSandbox`, which offers no protection from malicious code. Instead, if you have Podman/Docker, you can try using the `ContainerSandbox` instead, or follow [jonppe's instructions](https://github.com/jonppe/funsearch/blob/745f2e7a61ef1418a95e09a009f2f65a3ce7c2ac/README.md) to set up a container to run `funsearch` in.
- The python-script `examples/gasoline_spec.py` specifies the problem, evaluation-function, and the function to start search at.
- The parameter `11` is the parameter passed to the `evaluate` function in `examples/gasoline_spec.py`, i.e. the number of entries the `xs`-vector and `ys`-vector should have.

## Inspecting Found Functions

Eventually abort the search with ctrl+c. Inspect found functions via:

```shell
funsearch ls data/backups/program_db_gasoline_1731323471_0.pickle

# Example output:
# Found 10 programs
# 0: Program with score 1.2222222222222223
# def gasoline(xs: List[int], ys: List[int]) -> tuple[int, int]:
#   """Given a gasoline-problem specified by the list of x-values and y-values,
#   return a new gasoline-problem, with one additional x-value and y-value.
#   The integers are always non-negative.
#   """
#   x = np.random.randint(1, 10)  # generate a random x-value between 1 and 10
#   y = np.random.randint(1, 10)  # generate a random y-value between 1 and 10
#   return x, y
# 
# 
# 1: Program with score 1.2
# def gasoline(xs: List[int], ys: List[int]) -> tuple[int, int]:
#   """Given a gasoline-problem specified by the list of x-values and y-values,
#   return a new gasoline-problem, with one additional x-value and y-value.
#   The integers are always non-negative.
#   """
#   # Start by copying the code from gasoline_v0
#   x = [10, 18, 16, 5, 5][len(xs) % 5]
#   y = [6, 17, 10, 18, 15][len(ys) % 5]
# 
#   # Make small code changes to improve the function
#   # For example, we can increase the range of numbers for x and y
#   x = [10, 18, 16, 5, 5, 10, 15, 20][len(xs) % 8]
#   y = [6, 17, 10, 18, 15, 8, 12, 16][len(ys) % 8]
# 
#   return x, y
# 
# 
# 2: Program with score 1.105263157894737
# def gasoline(xs: List[int], ys: List[int]) -> tuple[int, int]:
#   """Given a gasoline-problem specified by the list of x-values and y-values,
#   return a new gasoline-problem, with one additional x-value and y-value.
#   The integers are always non-negative.
#   """
#   x = [7, 12, 9, 13, 14][len(xs) % 5]
#   y = [8, 11, 5, 19, 8][len(ys) % 5]
#   return x, y
# 
# 
# 3: Program with score 1.0
# def gasoline(xs: List[int], ys: List[int]) -> tuple[int, int]:
#   """Given a gasoline-problem specified by the list of x-values and y-values,
#   return a new gasoline-problem, with one additional x-value and y-value.
#   The integers are always non-negative.
#   """
#   x = [10, 18, 16, 5, 5][len(xs) % 5]
#   y = [6, 17, 10, 18, 15][len(ys) % 5]
#   return x, y
# 
# 
# 4: Program with score 1.0 [...]
```

---

Publication accompanying the [original repo](https://github.com/google-deepmind/funsearch):

> Romera-Paredes, B. et al. [Mathematical discoveries from program search with large language models]y(https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023)