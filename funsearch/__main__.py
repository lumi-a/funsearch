from __future__ import annotations

import json
import logging
import os
import pathlib
import pickle
import re
import time
from pathlib import Path

import click
import llm
from dotenv import load_dotenv

from funsearch import (
  code_manipulation,
  config,
  core,
  evaluator,
  sampler,
  sandbox,
)
from funsearch.programs_database import ProgramsDatabase

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)


def get_all_subclasses(cls):
  all_subclasses = []

  for subclass in cls.__subclasses__():
    all_subclasses.append(subclass)
    all_subclasses.extend(get_all_subclasses(subclass))

  return all_subclasses


SANDBOX_TYPES = [*get_all_subclasses(sandbox.DummySandbox), sandbox.DummySandbox]
SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]


def parse_input(filename_or_data: str):
  if len(filename_or_data) == 0:
    msg = "No input data specified"
    raise Exception(msg)
  p = pathlib.Path(filename_or_data)
  if p.exists():
    if p.name.endswith(".json"):
      return json.load(open(filename_or_data))
    if p.name.endswith(".pickle"):
      return pickle.load(open(filename_or_data, "rb"))
    msg = "Unknown file format or filename"
    raise Exception(msg)
  data = [filename_or_data] if "," not in filename_or_data else filename_or_data.split(",")
  if data[0].isnumeric():
    f = int if data[0].isdecimal() else float
    data = [f(v) for v in data]
  return data


def _most_recent_backup() -> Path:
  """Returns the most recent backup file."""

  # Define the directory and file pattern
  backup_dir = Path("data/backups")
  file_pattern = re.compile(r"program_db_.*_(\d+)_(\d+)\.pickle")

  # Find all matching files and extract (X, Y) values
  matching_files: list[tuple[int, int, Path]] = []
  for file in backup_dir.glob("program_db_*.pickle"):
    match = file_pattern.match(file.name)
    if match:
      timestamp, id = map(int, match.groups())
      matching_files.append((timestamp, id, file))

  # Select the file with lexicographically maximal (X, Y)
  if not matching_files:
    msg = "No matching backup files found in data/backups/"
    raise FileNotFoundError(msg)
  _, _, selected_file = max(matching_files)

  return selected_file


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
  pass


@main.command()
@click.argument("spec_file", type=click.File("r"))
@click.argument("inputs")
@click.option("--model_name", default="gpt-3.5-turbo", help="LLM model")
@click.option(
  "--output_path",
  default="./data/",
  type=click.Path(file_okay=False),
  help="path for logs and data",
)
@click.option("--iterations", default=-1, type=click.INT, help="Max iterations per sampler")
@click.option("--samplers", default=15, type=click.INT, help="Samplers")
@click.option(
  "--sandbox_type",
  default="ExternalProcessSandbox",
  type=click.Choice(SANDBOX_NAMES),
  help="Sandbox type",
)
def run(
  spec_file: click.File,
  inputs: str,
  model_name: str,
  output_path: click.Path,
  iterations: int,
  samplers: int,
  sandbox_type: str,
) -> None:
  r"""Execute FunSearch algorithm.

  \b
    SPEC_FILE is a python module that provides the basis of the LLM prompt as
              well as the evaluation metric.
              See specs/cap-set.py for an example.\n
  \b
    INPUTS    input filename ending in .json or .pickle, or a comma-separated
              input data. The files are expected to contain a list with at least
              one element. Elements shall be passed to the solve() method
              one by one. Examples
                8
                8,9,10
                ./specs/cap_set_input_data.json
  """
  load_dotenv()

  timestamp = str(int(time.time()))
  log_path = pathlib.Path(output_path) / timestamp
  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")

  model = llm.get_model(model_name)
  model.key = model.get_key()
  lm = sampler.LLM(2, model, log_path)

  specification = spec_file.read()
  function_to_evolve, function_to_run = core._extract_function_names(specification)
  template = code_manipulation.text_to_program(specification)

  conf = config.Config(num_evaluators=1)
  database = ProgramsDatabase(conf.programs_database, template, function_to_evolve, identifier=timestamp)

  inputs = parse_input(inputs)

  sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)

  def construct_evaluators(sampler_ix: int) -> list[evaluator.Evaluator]:
    evaluators: list[evaluator.Evaluator] = [
      evaluator.Evaluator(
        database,
        sandbox_class(base_path=log_path / f"sampler-{sampler_ix}-evaluator-{i}", timeout_secs=30),
        template,
        function_to_evolve,
        function_to_run,
        inputs,
      )
      for i in range(conf.num_evaluators)
    ]

    # We send the initial implementation to be analysed by one of the evaluators.
    # TODO: Only do this once for one evaluator
    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None)
    assert len(database._islands[0]._clusters) > 0, (
      "Initial analysis failed. Make sure that Sandbox works! See e.g. the error files under sandbox data."
    )

    return evaluators

  samplers = [sampler.Sampler(database, construct_evaluators(i), lm) for i in range(samplers)]
  core.run(samplers, database, iterations)


@main.command()
@click.argument("db_file", type=click.File("rb"), required=False)
def resume(db_file: click.File | None) -> None:
  """Continue running FunSearch from a backup.

  If not provided, selects the most recent one from data/backups/.
  """
  if db_file is None:
    db_file = _most_recent_backup().open("rb")

  database = ProgramsDatabase.load(db_file)

  core.run(samplers, database, iterations)


@main.command()
@click.argument("db_file", type=click.File("rb"), required=False)
def ls(db_file: click.File | None) -> None:
  """List programs from a stored database.

  If not provided, selects the most recent one from data/backups/.
  """
  if db_file is None:
    db_file = _most_recent_backup().open("rb")

  # Load and process the database
  conf = config.Config(num_evaluators=1)

  # TODO: Have ProgramsDatabase also include config and other parameters
  # TODO: Also put success-counts and as many other attributes in there
  database = ProgramsDatabase(conf.programs_database, None, "", identifier="")
  database.load(db_file)

  progs = database.get_best_programs_per_island()
  print(f"# Found {len(progs)} programs")  # noqa: T201
  for ix, (prog, score) in enumerate(reversed(progs)):
    i = len(progs) - 1 - ix
    print(f"# {i}: Program with score {score}")  # noqa: T201
    prog.name += f"_{i}"
    print(prog)  # noqa: T201
    print("\n")  # noqa: T201
  print(f"# Programs loaded from file: {db_file.name}")  # noqa: T201


if __name__ == "__main__":
  main()
