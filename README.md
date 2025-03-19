[FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/) uses evolutionary search on python-code to find better functions. Because randomly changing the characters in python-code makes it unrunnable, we instead ask an LLM to slightly change the code. See [examples](https://lumi-a.github.io/funsearch) of functions found by Funsearch.

## Setup

This project uses [uv](https://docs.astral.sh/uv/), which automatically installs python and dependencies. This project currently uses Mistral's LLMs. Set your Mistral-API-key in the environment-variable `MISTRAL_API_KEY`, or create a file named [`.env`](https://pypi.org/project/python-dotenv/) with the content:
```bash
MISTRAL_API_KEY=ABCDEFG
```
and replace `ABCDEFG` with your actual API-key.


## Running

Search for graphs on 35 vertices that have many edges but no 3-cycles or 4-cycles:

```sh
uv run funsearch start specs/3-4-cyclefree.py 35
```

- The code generated by the LLM is executed in a sandbox, which defaults to running the code on your system and offers no protection from malicious code. If you have Docker or Podman, you can try passing `--sandbox-type ContainerSandbox` for safer sandboxing, see [jonppe's instructions](https://github.com/jonppe/funsearch/blob/745f2e7a61ef1418a95e09a009f2f65a3ce7c2ac/README.md).
- The LLM defaults to gpt-3.5-turbo and can be changed with `--llm`. See `uv run llm models list` for available models.

See `uv run funsearch start --help` for more options.


## Inspecting Found Functions

Eventually abort the search with `ctrl+c`. Print the best functions:

```sh
# Inspect most recent backup:
uv run funsearch ls
# Inspect specific backup:
uv run funsearch ls data/backups/3-4-cyclefree_1731323471_0.pickle
```

## Tests
Run tests:

```sh
uv run pytest
```

Due to monkeypatching, this shouldn't execute any LLM-queries.

---

This is a fork of [jonppe's fork](https://github.com/jonppe/funsearch) of [google-deepmind's funsearch](https://github.com/google-deepmind/funsearch), the latter accompanied by the publication:

> Romera-Paredes, B. et al. [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023)