"""Generate json-files from funsearch output."""  # noqa: INP001

import json
import re
from pathlib import Path

from funsearch.programs_database import ProgramsDatabase

BACKUP_DIR = Path("../data/backups")
JSON_DIR = Path("json-data")  # Also set this in script.js

file_pattern = re.compile(r"(.*)_(\d+)_(\d+)\.pickle")

# For each tuple of (function_name, timestamp), only keep
# the (idx, path) one with the highest idx.
files: dict[tuple[str, int], tuple[int, Path]] = {}
for f in BACKUP_DIR.glob("*.pickle"):
  match = file_pattern.match(f.name)
  if match:
    specname, timestamp, idx = match.groups()
    try:
      timestamp, idx = int(timestamp), int(idx)
    except ValueError:
      continue
    if (specname, timestamp) not in files or idx > files[(specname, timestamp)][0]:
      files[(specname, timestamp)] = (idx, f)


def _to_filename(function_name: str, timestamp: int) -> Path:
  return JSON_DIR / f"{function_name}_{timestamp}.json"


for (specname, timestamp), (idx, file) in files.items():
  database = ProgramsDatabase.load(file.open("rb"))

  with _to_filename(specname, timestamp).open("w") as f:
    # As backups are indexed with timestamps, and we don't expect backups to change over time,
    # keep the json minimal, without newlines (which otherwise would be neat for VCS)
    json.dump(
      {
        "config": vars(database._config),  # noqa: SLF001
        "inputs": database.inputs,
        "specCode": database._specification,  # noqa: SLF001
        "problemName": database.problem_name,
        "timestamp": database.timestamp,
        "islands": [
          {
            "runs": island._runs,  # noqa: SLF001
            "improvements": island._improvements,  # noqa: SLF001
          }
          for island in database._islands  # noqa: SLF001
        ],
      },
      f,
      separators=(",", ":"),
    )

# Create the index of all json-files
with (JSON_DIR / "index.json").open("w") as f:
  json.dump(
    sorted([_to_filename(function_name, timestamp).name for (function_name, timestamp) in files]), f, indent=2
  )
