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
import threading
import time
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from absl import logging

from funsearch import code_manipulation
from funsearch import config as config_lib
from funsearch.evaluator import Evaluator
from funsearch.sandbox import ExternalProcessSandbox

Signature = tuple[float, ...]
ScoresPerTest = Mapping[float | str, float]


def extract_function_names(specification: str) -> tuple[str, str]:
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


import dataclasses


@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    functions_per_prompt: Number of previous programs to include in prompts.
    num_islands: Number of islands to maintain as a diversity mechanism.
    reset_period: How often (in seconds) the weakest islands should be reset.
    cluster_sampling_temperature_init: Initial temperature for softmax sampling
        of clusters within an island.
    cluster_sampling_temperature_period: Period of linear decay of the cluster
        sampling temperature.
    backup_period: Number of iterations before backing up the program database on disk
    backup_folder: Path for automatic backups

  """

  inputs: list[float] | list[str]
  specification: str
  problem_name: str
  message: str = ""
  functions_per_prompt: int = 2
  num_islands: int = 10
  reset_period: int = 4 * 60 * 60
  cluster_sampling_temperature_init: float = 0.1
  cluster_sampling_temperature_period: int = 30_000


class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
    self,
    config: ProgramsDatabaseConfig,
  ) -> None:
    self._config: ProgramsDatabaseConfig = config

    function_to_evolve, function_to_run = extract_function_names(config.specification)
    self._function_to_evolve: str = function_to_evolve
    self._function_to_run: str = function_to_run
    self._template: code_manipulation.Program = code_manipulation.text_to_program(config.specification)

    self._islands: list[Island] = [
      Island(
        self._template,
        function_to_evolve,
        config.functions_per_prompt,
        config.cluster_sampling_temperature_init,
        config.cluster_sampling_temperature_period,
      )
      for _ in range(config.num_islands)
    ]

    self._last_reset_time: float = time.time()

  def get_best_programs_per_island(self) -> Iterable[tuple[code_manipulation.Function | None, float]]:
    """Returns the best programs per island, together with their scores."""
    return sorted(
      [(island._best_program, island._best_score) for island in self._islands],
      key=lambda t: t[1],
      reverse=True,
    )

  def populate(self, log_path: pathlib.Path) -> bool:
    """Populate islands with the seed-function and return whether the seed-function ran successfully."""
    evaluator = Evaluator(
      ExternalProcessSandbox(log_path),
      self._template,
      self._function_to_evolve,
      self._function_to_run,
      self._config.inputs,
    )
    initial_sample = self._template.get_function(self._function_to_evolve).body
    program, scores_per_test = evaluator.analyse(initial_sample, version_generated=None, index=-1)
    if scores_per_test:
      for island_id in range(len(self._islands)):
        self._islands[island_id].register_program(program, scores_per_test)
      return True
    return False

  @classmethod
  def load(cls, file) -> ProgramsDatabase:
    """Load previously saved database."""
    data = pickle.load(file)
    database = ProgramsDatabase(config=ProgramsDatabaseConfig(**data["_config"]))
    for key in data:
      if key != "_config":
        setattr(database, key, data[key])

    return database

  def save(self, file) -> None:
    """Save database to a file."""
    data = {}
    keys = [
      "_config",
      "_islands",
      "_last_reset_time",
    ]
    for key in keys:
      data[key] = getattr(self, key)
    pickle.dump(data, file)

  def backup(self, backup_file: pathlib.Path) -> None:
    """Save a backup of the database to a backup-file."""
    with backup_file.open("wb") as f:
      self.save(f)

  def get_prompt(self) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    island_id = np.random.randint(len(self._islands))
    code, version_generated = self._islands[island_id].get_prompt()
    return Prompt(code, version_generated, island_id)

  def register_program(
    self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest
  ) -> None:
    """Registers `program` in the database."""
    self._islands[island_id].register_program(program, island_id, scores_per_test)

    # Check whether it is time to reset an island.
    # TODO: Move this to core.run or something
    if time.time() - self._last_reset_time > self._config.reset_period:
      self._last_reset_time = time.time()
      self._reset_islands()

  def register_failure(self, island_id: int) -> None:
    """Registers a failure on an island."""
    self._islands[island_id].register_failure()

  def _reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    num_islands = len(self._islands)
    indices_sorted_by_score: list[int] = sorted(
      range(num_islands),
      key=lambda ix: self._islands[ix]._best_score + np.random.random() * 1e-6,
    )
    num_islands_to_reset = num_islands // 2
    reset_islands_ids: list[int] = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids: list[int] = indices_sorted_by_score[num_islands_to_reset:]
    founders = [
      (self._islands[island_id]._best_program, self._islands[island_id]._best_scores_per_test)
      for island_id in keep_islands_ids
    ]
    for island_id in reset_islands_ids:
      self._islands[island_id] = Island(
        self._template,
        self._function_to_evolve,
        self._config.functions_per_prompt,
        self._config.cluster_sampling_temperature_init,
        self._config.cluster_sampling_temperature_period,
      )
      (program, scores_per_test) = np.random.choice(founders)
      self._islands[island_id].register_program(program, island_id, scores_per_test)

  def print_status(self) -> None:
    """Prints the current status of the database."""
    max_score = max(island._best_score for island in self._islands)
    # Subtract 1 due to the initial .populate() calls
    total_successes = sum(island._success_count - 1 for island in self._islands)  # noqa: SLF001
    total_failures = sum(island._failure_count for island in self._islands)  # noqa: SLF001
    attempts = total_successes + total_failures
    failure_rate = round(100 * total_failures / attempts if attempts > 0 else 0.0)

    print(f"Max-Score {max_score:8.3f} │ {attempts} samples │ {failure_rate}% failed")  # noqa: T201


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
    """Initialise island."""
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
    # This is (run_id, program).
    self._improvements: list[tuple[int, code_manipulation.Function]] = []

    # TODO: Initialise island with initial programs instead
    self._best_score: float = -float("inf")
    self._best_program: code_manipulation.Function | None = None
    self._best_scores_per_test: ScoresPerTest | None = None

    self._clusters: dict[Signature, Cluster] = {}
    self._num_programs_peroidic: int = 0

  def register_program(self, program: code_manipulation.Function, scores_per_test: ScoresPerTest) -> None:
    """Register a `program` on this island with a given `scores_per_test`."""
    signature = _get_signature(scores_per_test)
    if signature not in self._clusters:
      score = _reduce_score(scores_per_test)
      self._clusters[signature] = Cluster(score, program)
    else:
      self._clusters[signature].register_program(program)
    self._num_programs_peroidic += 1

    score = _reduce_score(scores_per_test)
    self.register_success(score)
    if score > self._best_score:
      self.register_improvement(program, score, scores_per_test)

  def register_improvement(
    self, program: code_manipulation.Function, score: float, scores_per_test: ScoresPerTest
  ) -> None:
    """Register the program that caused the latest improvement."""
    self._best_program = program
    self._best_score = score
    self._best_scores_per_test = scores_per_test
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
      renamed_implementation = code_manipulation.rename_function_calls(
        str(implementation), self._function_to_evolve, new_function_name
      )
      versioned_functions.append(code_manipulation.text_to_function(renamed_implementation))

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
    """A cluster, initialised with a single implementation and score."""
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
