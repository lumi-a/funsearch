# FunSearch

Fork of [jonppe's fork](https://github.com/jonppe/funsearch) of [google-deepmind's funsearch](https://github.com/google-deepmind/funsearch).

## Installation

```shell
python -m venv .venv        # Create virtual environment
source .venv/bin/activate   # On Windows, use `.venv\Scripts\activate`
                            # Use `deactivate` to exit the environment

pip install pdm             # For package- and dependency-management
pdm install --no-self       # Install project's dependencies
pip install -e --no-deps .  # Install project
mkdir data                  # Create directory for storing data
```

## Running

Consult `llm keys set --help` and set your API-key. The following runs funsearch with OpenAI's `gpt-3.5-turbo`:

```shell
funsearch run --model_name gpt-3.5-turbo --output_path data --sandbox_type ExternalProcessSandbox examples/gasoline_spec.py 11
```
<!-- TODO: Add example output -->


Explaining the arguments (run `funsearch run --help` for more options):

- Pick your `--model_name` from the list of available models (`llm models list`). To use other models, see the [llm-docs](https://llm.datasette.io/en/stable/other-models.html).
- The `--output_path` directory stores API-calls and responses.
- The `--sandbox_type` sets the sandbox in which the searched functions will be run. This might be desirable because, for evaluation, we execute python-code that the LLM gave to us without restrictions. The documentation of `/funsearch/sandbox.py` explains the different sandbox types. Above, we used `ExternalProcessSandbox`, which offers no protection from malicious code. Instead, if you have Podman/Docker, you can try using the `ContainerSandbox` instead, or follow [jonppe's instructions](https://github.com/jonppe/funsearch/blob/745f2e7a61ef1418a95e09a009f2f65a3ce7c2ac/README.md) to set up a container to run `funsearch` in.
- The python-script `examples/gasoline_spec.py` specifies the problem, evaluation-function, and the function to start search at.
- The parameter `11` is the parameter passed to the `evaluate` function in `examples/gasoline_spec.py`, i.e. the number of entries the `xs`-vector and `ys`-vector should have.

## Inspecting Found Functions

Eventually abort the search with ctrl+c. Inspect found functions via:

```shell
funsearch ls data/backups/program_db_gasoline_{timestamp}_0.pickle
```
<!-- TODO: Add example output -->

---

Publication accompanying the [original repo](https://github.com/google-deepmind/funsearch):

> Romera-Paredes, B. et al. [Mathematical discoveries from program search with large language models]y(https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023)