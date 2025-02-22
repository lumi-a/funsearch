# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""

from __future__ import annotations

import copy
import dataclasses
import pathlib
import pickle
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from absl import logging

from funsearch import code_manipulation
from funsearch import config as config_lib

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(code_manipulation.yield_decorated(specification, "funsearch", "run"))
  if len(run_functions) != 1:
    msg = "Expected 1 function decorated with `@funsearch.run`."
    raise ValueError(msg)
  evolve_functions = list(code_manipulation.yield_decorated(specification, "funsearch", "evolve"))
  if len(evolve_functions) != 1:
    msg = "Expected 1 function decorated with `@funsearch.evolve`."
    raise ValueError(msg)
  return evolve_functions[0], run_functions[0]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    msg = f"`logits` contains non-finite value(s): {non_finites}"
    raise ValueError(msg)
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  # Custom softmax to avoid scipy-dependency
  scaled = logits / temperature
  max_index = np.argmax(scaled, axis=-1)
  exp_scaled = np.exp(scaled - scaled[max_index])
  result = exp_scaled / np.sum(exp_scaled, axis=-1)

  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  result[max_index] = 1 - np.sum(result[0:max_index]) - np.sum(result[max_index + 1 :])
  return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score."""
  return scores_per_test[list(scores_per_test.keys())[-1]]


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
  """Represents test scores as a canonical signature."""
  return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.

  """

  code: str
  version_generated: int
  island_id: int


class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
    self,
    config: config_lib.ProgramsDatabaseConfig,
    specification: any,
    inputs: list[float | int] | list[str],
    problem_name: str,
    timestamp: int,
    message: str,
  ) -> None:
    self._config: config_lib.ProgramsDatabaseConfig = config
    self.inputs = inputs

    self._specification = specification
    function_to_evolve, function_to_run = _extract_function_names(specification)
    self.function_to_evolve: str = function_to_evolve
    self.function_to_run: str = function_to_run
    self.template: code_manipulation.Program = code_manipulation.text_to_program(specification)

    # Initialize empty islands.
    self._islands: list[Island] = []
    for _ in range(config.num_islands):
      self._islands.append(
        Island(
          self.template,
          function_to_evolve,
          config.functions_per_prompt,
          config.cluster_sampling_temperature_init,
          config.cluster_sampling_temperature_period,
        )
      )

    # TODO: Why not move these to Island?
    self._best_score_per_island: list[float] = [-float("inf")] * config.num_islands
    self._best_program_per_island: list[code_manipulation.Function | None] = [None] * config.num_islands
    self._best_scores_per_test_per_island: list[ScoresPerTest | None] = [None] * config.num_islands

    self._last_reset_time: float = time.time()
    self._program_counter = 0
    self._backups_done = 0

    self.problem_name = problem_name
    self.timestamp = timestamp
    self.message = message

  def get_best_programs_per_island(self) -> Iterable[tuple[code_manipulation.Function | None, float]]:
    return sorted(
      zip(self._best_program_per_island, self._best_score_per_island), key=lambda t: t[1], reverse=True
    )

  def save(self, file) -> None:
    """Save database to a file."""
    data = {}
    keys = [
      "_config",
      "inputs",
      "_specification",
      "_islands",
      "_best_score_per_island",
      "_best_program_per_island",
      "_best_scores_per_test_per_island",
      "_last_reset_time",
      "_program_counter",
      "_backups_done",
      "problem_name",
      "timestamp",
      "message",
    ]
    for key in keys:
      data[key] = getattr(self, key)
    pickle.dump(data, file)

  def load(file) -> ProgramsDatabase:
    """Load previously saved database."""
    data = pickle.load(file)

    database = ProgramsDatabase(
      config=data["_config"],
      specification=data["_specification"],
      inputs=data["inputs"],
      problem_name=data["problem_name"],
      timestamp=int(time.time()),
      message=data["message"],
    )

    for key in data:
      setattr(database, key, data[key])

    return database

  def backup(self) -> None:
    filename = f"{self.problem_name}_{self.timestamp}_{self._backups_done}.pickle"
    p = pathlib.Path(self._config.backup_folder)
    if not p.exists():
      p.mkdir(parents=True, exist_ok=True)
    filepath = p / filename
    logging.info(f"Saving backup to {filepath}")

    with open(filepath, mode="wb") as f:
      self.save(f)
    self._backups_done += 1

  def get_prompt(self) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    island_id = np.random.randint(len(self._islands))
    code, version_generated = self._islands[island_id].get_prompt()
    return Prompt(code, version_generated, island_id)

  def _register_program_in_island(
    self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest
  ) -> None:
    """Registers `program` in the specified island."""
    self._islands[island_id].register_program(program, scores_per_test)
    score = _reduce_score(scores_per_test)
    self._islands[island_id].register_success(score)
    if score > self._best_score_per_island[island_id]:
      self._best_program_per_island[island_id] = program
      self._best_scores_per_test_per_island[island_id] = scores_per_test
      self._best_score_per_island[island_id] = score
      self._islands[island_id].register_improvement(program)
      logging.info("✔ Best score of island %d increased to %s", island_id, score)

  def register_program(
    self, program: code_manipulation.Function, island_id: int | None, scores_per_test: ScoresPerTest
  ) -> None:
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    if island_id is None:
      # This is a program added at the beginning, so adding it to all islands.
      for island_id in range(len(self._islands)):
        self._register_program_in_island(program, island_id, scores_per_test)
    else:
      self._register_program_in_island(program, island_id, scores_per_test)

    # Check whether it is time to reset an island.
    if time.time() - self._last_reset_time > self._config.reset_period:
      self._last_reset_time = time.time()
      self.reset_islands()

    # Backup every N iterations
    if self._program_counter > 0:
      self._program_counter += 1
      if self._program_counter > self._config.backup_period:
        self._program_counter = 0
        self.backup()

  def register_failure(self, island_id: int) -> None:
    """Registers a failure in the database."""

    if island_id is not None:
      self._islands[island_id].register_failure()

  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    indices_sorted_by_score: np.ndarray = np.argsort(
      self._best_score_per_island + np.random.randn(len(self._best_score_per_island)) * 1e-6
    )
    num_islands_to_reset = self._config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    for island_id in reset_islands_ids:
      self._islands[island_id] = Island(
        self.template,
        self.function_to_evolve,
        self._config.functions_per_prompt,
        self._config.cluster_sampling_temperature_init,
        self._config.cluster_sampling_temperature_period,
      )
      self._best_score_per_island[island_id] = -float("inf")
      founder_island_id = np.random.choice(keep_islands_ids)
      founder = self._best_program_per_island[founder_island_id]
      founder_scores = self._best_scores_per_test_per_island[founder_island_id]
      self._register_program_in_island(founder, island_id, founder_scores)

  def log_tabular(self, last_run: bool) -> None:
    scores = self._best_score_per_island
    score_width = max(5, *(len(str(x)) for x in scores))
    separator = "  "

    headers = [
      f"{'Isl':>3}",
      f"{'Score':>{score_width}}",
      f"{'Queries':>7}",
      f"{'Failures':>8}",
      f"{'ok%':>3}",
    ]
    output = [separator.join(headers), separator.join("─" * len(x) for x in headers)]

    total_successes = 0
    total_failures = 0
    for island_idx, score in sorted(enumerate(scores), key=lambda t: t[1], reverse=True):
      # Subtract one to account for the program that was added at the beginning.
      successes = self._islands[island_idx]._success_count - 1  # noqa: SLF001
      failures = self._islands[island_idx]._failure_count  # noqa: SLF001
      total_successes += successes
      total_failures += failures

      attempts = successes + failures
      success_rate = int(100 * successes / attempts if attempts > 0 else 0)

      columns = [
        f"{island_idx:>3}",
        f"{score:>{score_width}}",
        f"{attempts:>7}",
        f"{failures:>8}",
        f"{success_rate:>2.0f}% ",
      ]
      output.append(separator.join(columns))

    total_attempts = total_successes + total_failures
    total_success_rate = int(100 * total_successes / total_attempts if total_attempts > 0 else 0)
    summary = [
      f"{total_attempts:>7}",
      f"{total_failures:>8}",
      f"{total_success_rate:>2.0f}% ",
    ]
    output.append((" " * (2 * len(separator) + score_width - 5)) + " Total: " + separator.join(summary))
    print("\n".join(output))  # noqa: T201

    if not last_run:
      lines_to_move = len(output)
      print(f"\033[{lines_to_move}A", end="", flush=True)  # noqa: T201
      sys.stdout.flush()


class Island:
  """A sub-population of the programs database."""

  def __init__(
    self,
    template: code_manipulation.Program,
    function_to_evolve: str,
    functions_per_prompt: int,
    cluster_sampling_temperature_init: float,
    cluster_sampling_temperature_period: int,
  ) -> None:
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self._functions_per_prompt: int = functions_per_prompt
    self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self._cluster_sampling_temperature_period = cluster_sampling_temperature_period

    # The island-runs over time. None means a failure, otherwise a float representing the score.
    self._runs: list[float | None] = []
    self._success_count: int = 0  # This should always equal len([x for x in self._runs if x is not None])
    self._failure_count: int = 0  # This should always equal len([x for x in self._runs if x is None])
    # For each improvement, keep track of the program that caused the improvement.
    self._improvements: [tuple[int, code_manipulation.Function]] = []

    self._clusters: dict[Signature, Cluster] = {}
    self._num_programs_peroidic: int = 0

  def register_program(self, program: code_manipulation.Function, scores_per_test: ScoresPerTest) -> None:
    """Stores a program on this island, in its appropriate cluster."""
    signature = _get_signature(scores_per_test)
    if signature not in self._clusters:
      score = _reduce_score(scores_per_test)
      self._clusters[signature] = Cluster(score, program)
    else:
      self._clusters[signature].register_program(program)
    self._num_programs_peroidic += 1

  def register_improvement(self, program: code_manipulation.Function) -> None:
    """Register the program that caused the latest improvement."""
    self._improvements.append((len(self._runs) - 1, program))

  def register_failure(self) -> None:
    """Register a failure on this island."""
    self._runs.append(None)
    self._failure_count += 1

  def register_success(self, score: float) -> None:
    """Register a success on this island."""
    self._runs.append(score)
    self._success_count += 1

  def get_prompt(self) -> tuple[str, int]:
    """Constructs a prompt containing functions from this island."""
    signatures = list(self._clusters.keys())
    cluster_scores = np.array([self._clusters[signature].score for signature in signatures])

    # Convert scores to probabilities using softmax with temperature schedule.
    period = self._cluster_sampling_temperature_period
    temperature = self._cluster_sampling_temperature_init * (
      1 - (self._num_programs_peroidic % period) / period
    )
    probabilities = _softmax(cluster_scores, temperature)

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

    idx = np.random.choice(len(signatures), size=functions_per_prompt, p=probabilities)
    chosen_signatures = [signatures[i] for i in idx]
    implementations = []
    scores = []
    for signature in chosen_signatures:
      cluster = self._clusters[signature]
      implementations.append(cluster.sample_program())
      scores.append(cluster.score)

    indices = np.argsort(scores)
    sorted_implementations = [implementations[i] for i in indices]
    version_generated = len(sorted_implementations) + 1
    return self._generate_prompt(sorted_implementations), version_generated

  def _generate_prompt(self, implementations: Sequence[code_manipulation.Function]) -> str:
    """Creates a prompt containing a sequence of function `implementations`."""
    implementations = copy.deepcopy(implementations)  # We will mutate these.

    # Format the names and docstrings of functions to be included in the prompt.
    versioned_functions: list[code_manipulation.Function] = []
    for i, implementation in enumerate(implementations):
      new_function_name = f"{self._function_to_evolve}_v{i}"
      implementation.name = new_function_name
      # Update the docstring for all subsequent functions after `_v0`.
      if i >= 1:
        implementation.docstring = f"Improved version of `{self._function_to_evolve}_v{i - 1}`."
      # If the function is recursive, replace calls to itself with its new name.
      implementation = code_manipulation.rename_function_calls(
        str(implementation), self._function_to_evolve, new_function_name
      )
      versioned_functions.append(code_manipulation.text_to_function(implementation))

    # Create the header of the function to be generated by the LLM.
    next_version = len(implementations)
    new_function_name = f"{self._function_to_evolve}_v{next_version}"
    header = dataclasses.replace(
      implementations[-1],
      name=new_function_name,
      body="",
      docstring=(f"Improved version of `{self._function_to_evolve}_v{next_version - 1}`."),
    )
    versioned_functions.append(header)

    # Replace functions in the template with the list constructed here.
    prompt = dataclasses.replace(self._template, functions=versioned_functions)
    return str(prompt)


class Cluster:
  """A cluster of programs on the same island and with the same Signature."""

  def __init__(self, score: float, implementation: code_manipulation.Function) -> None:
    # TODO: This is never used?
    self._score = score
    self._programs: list[code_manipulation.Function] = [implementation]
    self._lengths: list[int] = [len(str(implementation))]

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  def register_program(self, program: code_manipulation.Function) -> None:
    """Adds `program` to the cluster."""
    self._programs.append(program)
    self._lengths.append(len(str(program)))

  def sample_program(self) -> code_manipulation.Function:
    """Samples a program, giving higher probability to shorther programs."""
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (max(self._lengths) + 1e-6)
    probabilities = _softmax(-normalized_lengths, temperature=1.0)
    return np.random.choice(self._programs, p=probabilities)
