import re
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from funsearch import __main__, sampler
from funsearch.programs_database import ProgramsDatabase

ROOT_DIR = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def dummy_setup(monkeypatch, tmp_path):  # noqa: ARG001
  """Replace LLM-calls and override the output path to a temporary directory."""

  # Replace LLM call
  monkeypatch.setattr(sampler.LLM, "draw_sample", lambda _self, _prompt, _ix: "  return 0.5")


@pytest.mark.filterwarnings(
  "ignore",
  message=r"The numpy\.linalg\.linalg has been made private and renamed to numpy\.linalg\._linalg\.",
  category=DeprecationWarning,
  module="numpy",
)
def test_cli(tmp_path: Path):
  runner = CliRunner()

  spec_file = ROOT_DIR / "tests" / "fixtures" / "cap-set.py"
  start_output_path = tmp_path / "data-start"
  start_backup_path = start_output_path / "backups"
  timestamp = time.time()
  message = "Test message"
  samples = 3
  functions_per_prompt = 2
  num_islands = 7
  reset_period = 5432
  cluster_sampling_temperature_init = 0.1234
  cluster_sampling_temperature_period = 1234

  start_result = runner.invoke(
    __main__.start,
    [
      str(spec_file),
      "3",
      message,
      "--output-path",
      str(start_output_path),
      "--islands",
      str(num_islands),
      "--samples",
      str(samples),
      "--functions-per-prompt",
      str(functions_per_prompt),
      "--reset-period",
      str(reset_period),
      "--cluster-sampling-temperature-init",
      str(cluster_sampling_temperature_init),
      "--cluster-sampling-temperature-period",
      str(cluster_sampling_temperature_period),
    ],
  )

  assert start_result.exit_code == 0, start_result.output

  # Check that a log directory was created.
  # The log directory is expected to be output_path / <spec_file stem> / <timestamp>
  log_dir = start_output_path / spec_file.stem
  assert log_dir.exists(), "Log directory was not created in the expected location"

  # Check a timestamped subdirectory was created with a timestamp not far from the
  # timestamp we queried earlier.
  def matches_timestamp(d: Path) -> bool:
    if d.is_dir():
      try:
        dir_timestamp = int(d.name)
        if abs(dir_timestamp - timestamp) < 2.0:
          return True
      except ValueError:
        pass
    return False

  assert any(map(matches_timestamp, log_dir.iterdir())), (
    "No timestamped subdirectory was created in the spec log directory"
  )

  dbfile = next(start_backup_path.iterdir())
  with dbfile.open("rb") as f:
    database = ProgramsDatabase.load(f)

  assert database._config.problem_name == spec_file.stem, "Problem name was not saved in Database"
  assert database._config.message == message, "Message was not saved in Database"
  assert database._config.inputs == [3], "Inputs were not saved in Database"
  assert database._config.specification == spec_file.read_text(), "Specification was not saved in Database"
  assert database._config.cluster_sampling_temperature_init == cluster_sampling_temperature_init, (
    "Temperature was not saved in Database"
  )
  assert database._config.cluster_sampling_temperature_period == cluster_sampling_temperature_period, (
    "Temperature was not saved in Database"
  )
  assert database._config.functions_per_prompt == functions_per_prompt, "Functions per prompt was not saved in Database"
  assert database._config.reset_period == reset_period, "Reset period was not saved in Database"

  # Subtract 1 due to the initial .populate() call
  sample_count = sum(len(island._runs) - 1 for island in database._islands)
  assert sample_count == samples, "The run didn't last the correct number of samples"

  #################
  # Try resuming

  resume_output_path = tmp_path / "data-resume"
  resume_result = runner.invoke(
    __main__.resume,
    [
      str(dbfile),
      "--output-path",
      str(resume_output_path),
      "--samples",
      str(samples),
    ],
  )

  assert resume_result.exit_code == 0, resume_result.output

  resume_dbfile = next((resume_output_path / "backups").iterdir())
  with resume_dbfile.open("rb") as f:
    resume_database = ProgramsDatabase.load(f)

  assert database._config == resume_database._config, "Configs should match"

  # Subtract 1 due to the initial .populate() call. Multiply by two because we resumed the previous run
  assert sum(len(island._runs) - 1 for island in resume_database._islands) == 2 * samples, "Incorrect number of samples"

  #############
  # Try listing

  ls_result = runner.invoke(__main__.ls, [str(resume_dbfile)])

  assert ls_result.exit_code == 0, ls_result.output

  assert f"# Found {num_islands} programs in file" in ls_result.output, "Expected string in output"

  num_found = len(
    re.findall(
      r'''# Island \d, score 8\.0:
def priority_\d\(el: tuple\[int, \.\.\.\], n: int\) -> float:
  """Returns the priority with which we want to add `element` to the cap set\.
  el is a tuple of length n with values 0-2\.
  """
  return 0\.0
''',
      ls_result.output,
    )
  )
  assert num_found == num_islands, "Expected functions in output"


def test_parse_input():
  assert __main__._parse_input("1") == [1]
  assert __main__._parse_input("1,2,3") == [1, 2, 3]
  assert __main__._parse_input(str(ROOT_DIR / "tests" / "fixtures" / "inputs-numeric.json")) == [9, 10, 11]
  assert __main__._parse_input(str(ROOT_DIR / "tests" / "fixtures" / "inputs-string.json")) == ["a", "bc", "def"]
