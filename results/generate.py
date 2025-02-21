"""Generate json-files from funsearch output."""  # noqa: INP001

import json
from pathlib import Path
import re

from funsearch import config
from funsearch.code_manipulation import Function
from funsearch.programs_database import ProgramsDatabase

BACKUP_DIR = Path("../data/backups")
JSON_DIR = Path("json-data")

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

for (function_name, timestamp), (idx, file) in files.items():
  # TODO: This is a hack for now, we should read config and other params from the backup-database
  conf = config.Config(num_evaluators=1)
  database = ProgramsDatabase(conf.programs_database, None, "", identifier="")
  database.load(file.open("rb"))

  with (JSON_DIR / f"{function_name}_{timestamp}.json").open("w") as f:
    json.dump(
      [{"function": str(fn), "score": score} for (fn, score) in database.get_best_programs_per_island()], f
    )
