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

from funsearch import config, core
from funsearch.programs_database import ProgramsDatabase

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
  file_pattern = re.compile(r".*_(\d+)_(\d+)\.pickle")

  # Find all matching files and extract (X, Y) values
  matching_files: list[tuple[int, int, Path]] = []
  for file in backup_dir.glob("*.pickle"):
    match = file_pattern.match(file.name)
    if match:
      timestamp, idx = map(int, match.groups())
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
@click.argument("spec_file", type=click.File("r"))
@click.argument("inputs", type=_parse_input)
@click.argument("message", type=str, default="")
@click.option("--llm", default="gpt-3.5-turbo", type=click.Choice(MODELS), help="LLM")
@click.option(
  "--output-path", default="./data/", type=click.Path(file_okay=False), help="Path for logs and data"
)
@click.option("--iterations", default=-1, type=click.INT, help="Max iterations per sampler")
def start(
  spec_file: click.File,
  inputs: list[float] | list[str],
  message: str,
  llm: str,
  output_path: click.Path,
  iterations: int,
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
  conf = config.ProgramsDatabaseConfig(backup_folder=str(pathlib.Path(output_path) / "backups"))

  timestamp = str(int(time.time()))
  problem_name = Path(spec_file.name).stem

  database = ProgramsDatabase(conf, spec_file.read(), inputs, problem_name, timestamp, message)

  log_path = pathlib.Path(output_path) / problem_name / timestamp
  log_path.mkdir(exist_ok=True, parents=True)
  if not database.populate(log_path):
    msg = "Running initial function failed, see logs in output_path"
    raise RuntimeError(msg)
  core.run(database, llm, log_path, iterations)


@main.command(context_settings={"show_default": True})
@click.argument("db_file", type=click.File("rb"), required=False)
@click.option("--llm", default="gpt-3.5-turbo", type=click.Choice(MODELS), help="LLM")
@click.option(
  "--output-path", default="./data/", type=click.Path(file_okay=False), help="Path for logs and data"
)
@click.option("--iterations", default=-1, type=click.INT, help="Max iterations per sampler")
def resume(db_file: click.File | None, llm: str, output_path: click.Path, iterations: int) -> None:
  """Continue running FunSearch from a backup.

  If not provided, selects the most recent one from data/backups/.
  """
  if db_file is None:
    db_file = _most_recent_backup().open("rb")
  database = ProgramsDatabase.load(db_file)
  # TODO: There has to be a better way than doing this.
  # Maybe require backup-path as an argument?
  old_config = database._config.__dict__
  old_config["backup_folder"] = str(pathlib.Path(output_path) / "backups")
  database._config = config.ProgramsDatabaseConfig(**old_config)

  timestamp = str(int(time.time()))
  database.timestamp = timestampthiswontwork  # Database doesn't have attribute timestamp anymore

  log_path = pathlib.Path(output_path) / database._config.problem_name / timestamp
  log_path.mkdir(exist_ok=True, parents=True)

  core.run(database, llm, log_path, iterations)


@main.command()
@click.argument("db_file", type=click.File("rb"), required=False)
def ls(db_file: click.File | None) -> None:
  """List programs from a stored database.

  If not provided, selects the most recent one from data/backups/.
  """
  if db_file is None:
    db_file = _most_recent_backup().open("rb")

  database = ProgramsDatabase.load(db_file)

  progs = database.get_best_programs_per_island()
  print(f"# Found {len(progs)} programs")  # noqa: T201
  for ix, (prog, score) in enumerate(reversed(progs)):
    i = len(progs) - 1 - ix
    print(f"# {i}: Program with score {score}")  # noqa: T201
    prog.name += f"_{i}"
    print(prog)  # noqa: T201
    print("\n")  # noqa: T201
  print(f"# Programs loaded from file: {db_file.name}")  # noqa: T201


@main.command()
@click.argument("db_file", type=click.File("rb+"), required=False)
def change_db_message(db_file: click.File | None) -> None:
  """Change the message of a database.

  If not provided, selects the most recent one from data/backups/.
  """
  if db_file is None:
    db_file = _most_recent_backup().open("rb+")

  database = ProgramsDatabase.load(db_file)
  old_message = database.message

  database.message = click.edit(old_message, require_save=False)

  if database.message == old_message:
    click.echo("Database message unchanged.")
    db_file.close()
    return
  click.echo("Changed message from")
  click.echo(click.style(old_message, fg="blue"))
  click.echo("to")
  click.echo(click.style(database.message, fg="green"))

  db_file.seek(0)
  db_file.truncate()
  database.save(db_file)

  db_file.close()


if __name__ == "__main__":
  main()
