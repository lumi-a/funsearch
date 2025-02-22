"""Generate json-files from funsearch output.

This loads all databases in ./data/backups and converts them to json-files in the
./docs/json-data directory, where they can be read by ./docs/index.html.
After pushing the json-files to main, they are displayed on github-pages:

  https://lumi-a.github.io/funsearch

Neither the website nor this script claims to be robust, they're just
an easy "move things, break fast" way of sharing results with others.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from funsearch.programs_database import ProgramsDatabase

BACKUP_DIR = Path("data/backups")
JSON_DIR = Path("docs/json-data")  # Also set this in script.js

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
  return f"{function_name}_{timestamp}.json"


# Save small descriptions of each json-file in index.json
# Has schema (problemName, inputs, maxScore, message, timestamp, filepath)
index_json: list[tuple[str, list[float | int] | list[str], float, str, int, str]] = []
for (specname, timestamp), (idx, file) in files.items():
  database = ProgramsDatabase.load(file.open("rb"))

  with (JSON_DIR / _to_filename(specname, timestamp)).open("w") as f:
    # Trim message to 255 "characters"
    # This is bad practice, something something graphemes, but it will be cut off on the website anyway.
    index_json.append(
      (
        database.problem_name,
        database.inputs,
        max(database._best_score_per_island),  # noqa: SLF001
        database.message[:255],
        timestamp,
        _to_filename(specname, timestamp),
      )
    )

    # As backups are indexed with timestamps, and we don't expect backups to change over time,
    # keep the json minimal, without newlines (which otherwise would be neat for VCS)
    json.dump(
      {
        "problemName": database.problem_name,
        "inputs": database.inputs,
        "message": database.message,
        "config": vars(database._config),  # noqa: SLF001
        "specCode": database._specification,  # noqa: SLF001
        "timestamp": database.timestamp,
        "highestRunIndex": max(len(island._runs) for island in database._islands),  # noqa: SLF001
        "islands": [
          {
            "improvements": [(ix, island._runs[ix], str(program)) for ix, program in island._improvements],  # noqa: SLF001
            "successCount": island._success_count,  # noqa: SLF001
            "failureCount": island._failure_count,  # noqa: SLF001
          }
          for island in database._islands  # noqa: SLF001
        ],
      },
      f,
      separators=(",", ":"),
      indent=2,
    )

# Create the index of all json-files
with (JSON_DIR / "index.json").open("w") as f:
  json.dump(sorted(index_json), f, separators=(",", ":"), indent=2)
