"""Generate json-files from funsearch output."""  # noqa: INP001

import json
import re
from pathlib import Path

from funsearch import config
from funsearch.programs_database import ProgramsDatabase

BACKUP_DIR = Path("../data/backups")
JSON_DIR = Path("json-data")  # Also set this in script.js

file_pattern = re.compile(r"program_db_(.*)_(\d+)_(\d+)\.pickle")

# For each tuple of (function_name, timestamp), only keep
# the (idx, path) one with the highest idx.
files: dict[tuple[str, int], tuple[int, Path]] = {}
for f in BACKUP_DIR.glob("program_db_*.pickle"):
  match = file_pattern.match(f.name)
  if match:
    function_name, timestamp, idx = match.groups()
    try:
      timestamp, idx = int(timestamp), int(idx)
    except ValueError:
      continue
    if (function_name, timestamp) not in files or idx > files[(function_name, timestamp)][0]:
      files[(function_name, timestamp)] = (idx, f)


def _to_filename(function_name: str, timestamp: int) -> Path:
  return JSON_DIR / f"{function_name}_{timestamp}.json"


for (function_name, timestamp), (idx, file) in files.items():
  # TODO: This is a hack for now, we should read config and other params from the backup-database
  conf = config.Config(num_evaluators=1)
  database = ProgramsDatabase.load(file.open("rb"))

  with _to_filename(function_name, timestamp).open("w") as f:
    # As backups are indexed with timestamps, and we don't expect backups to change over time,
    # keep the json minimal, without newlines (which otherwise would be neat for VCS)
    json.dump(
      [{"fn": str(fn), "score": score} for (fn, score) in database.get_best_programs_per_island()],
      f,
      separators=(",", ":"),
    )
# Create the index of all json-files
with (JSON_DIR / "index.json").open("w") as f:
  json.dump(
    sorted([_to_filename(function_name, timestamp).name for (function_name, timestamp) in files]), f, indent=2
  )
