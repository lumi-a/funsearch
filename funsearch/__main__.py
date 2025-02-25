from __future__ import annotations

import copy
from dataclasses import replace
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

from funsearch import core
from funsearch.programs_database import ProgramsDatabase, ProgramsDatabaseConfig

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)


def _parse_input(filename_or_data: str) -> list[float] | list[str]:
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
  file_pattern = re.compile(r".*_(\d+)(|_\d+)\.pickle")

  # Find all matching files and extract (X, Y) values
  matching_files: list[tuple[int, str, Path]] = []
  for file in backup_dir.glob("*.pickle"):
    match = file_pattern.match(file.name)
    if match:
      timestamp, idx = int(match.group(1)), match.group(2)
      try:
        idx = int(idx)
      except ValueError:
        idx = 10000000000000000  # HACK: Rework.
      matching_files.append((timestamp, idx, file))

  # Select the file with lexicographically maximal (X, Y)
  if not matching_files:
    msg = "No matching backup files found in data/backups/"
    raise FileNotFoundError(msg)
  _, _, selected_file = max(matching_files)

  return selected_file


@click.group()
@click.pass_context
def main(_ctx: click.Context) -> None:
  pass


def _get_all_subclasses(cls: type) -> set[type]:
  return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in _get_all_subclasses(c)])


# TODO: Once click 8.2.0 releases, use better click.Choice
MODELS: list[str] = list(llm.get_model_aliases().keys())


@main.command(context_settings={"show_default": True})
@click.argument("spec_file", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path))
@click.argument("inputs", type=_parse_input)
@click.argument("message", type=str, default="")
@click.option("--llm", default="gpt-3.5-turbo", type=click.Choice(MODELS), help="LLM")
@click.option(
  "--output-path",
  default="./data/",
  type=click.Path(file_okay=False, path_type=Path),
  help="Path for logs and data",
)
@click.option("--samples", default=-1, type=click.INT, help="Maximum number of samples")
@click.option(
  "--functions-per-prompt", default=2, type=click.INT, help="How many past functions to send per prompt"
)
@click.option(
  "--islands", default=10, type=click.INT, help="How many islands (mostly separate populations) to use"
)
@click.option(
  "--reset-period",
  default=100_000,
  type=click.INT,
  help="Number of iterations before half the islands are reset",
)
@click.option(
  "--cluster-sampling-temperature-init",
  default=0.1,
  type=click.FLOAT,
  help="Initial temperature for cluster sampling",
)
@click.option(
  "--cluster-sampling-temperature-period",
  default=30_000,
  type=click.INT,
  help="Number of samples before temperature resets",
)
def start(
  spec_file: Path,
  llm: str,
  samples: int,
  output_path: Path,
  inputs: list[float] | list[str],
  message: str,
  functions_per_prompt: int,
  reset_period: int,
  islands: int,
  cluster_sampling_temperature_init: float,
  cluster_sampling_temperature_period: int,
) -> None:
  """Execute FunSearch algorithm.

  \b
    SPEC_FILE is a python module that provides the basis of the LLM prompt
              as well as the evaluation metric.
              See specs/cap-set.py for an example.\n
  \b
    INPUTS    is a filename ending in .json or .pickle, or comma-separated
              input data. The files are expected to contain a list with at
              least one element. Elements shall be passed to the solve()
              method one by one. Examples:
                8
                8,9,10
                ./specs/cap_set_input_data.json
  """  # noqa: D301
  timestamp = int(time.time())
  problem_name = spec_file.stem  # Not great, but it's not a pathlib-file.
  initial_log_path = output_path / problem_name / str(timestamp) / "_initial"
  config = ProgramsDatabaseConfig(
    inputs=inputs,
    specification=spec_file.read_text(),
    problem_name=problem_name,
    message=message,
    functions_per_prompt=functions_per_prompt,
    num_islands=islands,
    reset_period=reset_period,
    cluster_sampling_temperature_init=cluster_sampling_temperature_init,
    cluster_sampling_temperature_period=cluster_sampling_temperature_period,
  )
  database = ProgramsDatabase(config, initial_log_path)
  core.run(database, llm, output_path, timestamp, samples)


@main.command(context_settings={"show_default": True})
@click.argument("db_file", type=click.File("rb"), required=False)
@click.option("--llm", default="gpt-3.5-turbo", type=click.Choice(MODELS), help="LLM")
@click.option(
  "--output-path",
  default="./data/",
  type=click.Path(file_okay=False, path_type=Path),
  help="Path for logs and data",
)
@click.option("--samples", default=-1, type=click.INT, help="Maximum number of samples")
def resume(db_file: click.File | None, llm: str, output_path: click.Path, samples: int) -> None:
  """Continue running FunSearch from a backup.

  If not provided, selects the most recent one from data/backups/.
  """
  if db_file is None:
    db_file = _most_recent_backup().open("rb")
  database = ProgramsDatabase.load(db_file)

  core.run(database, llm, output_path, int(time.time()), samples)


@main.command()
@click.argument("db_file", type=click.File("rb"), required=False)
def ls(db_file: click.File | None) -> None:
  """List programs from a stored database.

  If not provided, selects the most recent one from data/backups/.
  """
  if db_file is None:
    db_file = _most_recent_backup().open("rb")
  database = ProgramsDatabase.load(db_file)

  def comment(string: str) -> None:
    click.echo(click.style(string, fg="green"))

  progs = list(database.get_best_programs_per_island())
  for ix, (program, score) in enumerate(reversed(progs)):
    i = len(progs) - 1 - ix
    comment(f"# Island {i}, score {score}:")
    renamed_program = copy.deepcopy(program)
    renamed_program.name += f"_{i}"
    click.echo(str(renamed_program))
  comment(f"# Found {len(progs)} programs in file: {db_file.name}")


@main.command()
@click.argument("db_file", type=click.File("rb+"), required=False)
def change_db_message(db_file: click.File | None) -> None:
  """Change the message of a database.

  If not provided, selects the most recent one from data/backups/.
  """
  if db_file is None:
    db_file = _most_recent_backup().open("rb+")

  database = ProgramsDatabase.load(db_file)
  old_message = database._config.message

  new_message = click.edit(old_message, require_save=False)

  if new_message == old_message:
    click.echo("Database message unchanged.")
    db_file.close()
    return
  click.echo("Changed message from")
  click.echo(click.style(old_message, fg="blue"))
  click.echo("to")
  click.echo(click.style(new_message, fg="green"))

  replace(database._config, message=new_message)

  db_file.seek(0)
  db_file.truncate()
  database.save(db_file)

  db_file.close()


if __name__ == "__main__":
  main()
