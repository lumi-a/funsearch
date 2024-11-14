# FunSearch

Fork of [jonppe's fork](https://github.com/jonppe/funsearch) of [google-deepmind's funsearch](https://github.com/google-deepmind/funsearch).

## Getting Started

```bash
python -m venv .venv  # Create virtual environment
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
# Use `deactivate` to exit the virtual environment

pip install pdm  # For package- and dependency-management
pdm install --no-self  # Install project's dependencies
pip install -e --no-deps .  # Install project
mkdir data  # Create directory for storing data
```

Consult `llm keys set --help` and set your API-key. The following runs funsearch with OpenAI's `gpt-3.5-turbo`:

```bash
funsearch run --model_name gpt-3.5-turbo --output_path ./data --sandbox_type ExternalProcessSandbox ./examples/gasoline_spec.py 11
```

Explanation for above arguments (run `funsearch run --help` for more options)

- List your available models via `llm models list`. To use other models, see the [llm-docs](https://llm.datasette.io/en/stable/other-models.html).
- The `--output_path` directory stores API-calls and responses.
- The `--sandbox_type` sets the sandbox in which the searched functions will be run. Since we execute code that the LLM gave to us without restrictions, you may want sandboxing. The documentation in `/funsearch/sandbox.py` explains the different sandbox types. The sandbox used above `ExternalProcessSandbox` offers no protection from malicious code. Instead, if you have Podman/Docker, you can try using the `ContainerSandbox` instead, or follow [jonppe's instructions](https://github.com/jonppe/funsearch/blob/745f2e7a61ef1418a95e09a009f2f65a3ce7c2ac/README.md) to set up a container to run `funsearch` in.
- In the python-script `./examples/gasoline_spec.py`, we specify the problem, evaluation-function, and initial function.
- The parameter `11` is the parameter passed to the `evaluate` function in `./examples/gasoline_spec.py`, i.e. the number of entries the `xs`-vector and `ys`-vector should have.

<!-- TODO: Add example output -->

---

Publication accompanying the [original repo]([google-deepmind's funsearch](https://github.com/google-deepmind/funsearch)):

> Romera-Paredes, B. et al. [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023)