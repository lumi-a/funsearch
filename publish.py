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


def publish() -> None:
    file_pattern = re.compile(r"(.*)_(\d+)\.pickle")

    # (function_name, timestamp) -> path
    files: dict[tuple[str, int], Path] = {}
    for f in BACKUP_DIR.glob("*.pickle"):
        match = file_pattern.match(f.name)
        if match:
            specname, timestamp = match.groups()
            files[(specname, int(timestamp))] = f

    # Save small descriptions of each json-file in index.json
    # Has schema (problemName, timestamp, inputs, maxScore, message, filepath)
    index_json: list[tuple[str, int, list[float] | list[str], float, str, str]] = []
    old_filenames = {file.name for file in JSON_DIR.glob("*json")}
    new_filenames = set()
    for (specname, timestamp), file in files.items():
        database = ProgramsDatabase.load(file.open("rb"))

        filename = f"{specname}_{timestamp}.json"
        new_filenames.add(filename)
        with (JSON_DIR / filename).open("w") as f:
            # Trim message to 255 "characters"
            # This is bad practice, something something graphemes, but it will be cut off on the website anyway.
            shortened_message = database._config.message.splitlines()[0][:255] if database._config.message != "" else ""
            index_json.append(
                (
                    database._config.problem_name,
                    timestamp,
                    database._config.inputs,
                    max(island._best_score for island in database._islands),
                    shortened_message,
                    filename,
                )
            )

            # For consistent interfacing with Javascript,
            json.dump(
                {
                    "config": vars(database._config),  # noqa: SLF001
                    "timestamp": timestamp,
                    "highestRunIndex": max(len(island._runs) for island in database._islands),  # noqa: SLF001
                    "islands": [
                        {
                            "improvements": [
                                (ix, island._runs[ix], str(program)) for ix, program in island._improvements
                            ],
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
    new_filenames.add("index.json")

    for filename in old_filenames - new_filenames:
        print(f"❌ Missing {filename}")
    for filename in new_filenames - old_filenames:
        print(f"✅ Added {filename}")


if __name__ == "__main__":
    publish()
