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

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

from funsearch import code_manipulation, programs_database
from funsearch.programs_database import Island, ProgramsDatabase, ProgramsDatabaseConfig, extract_function_names

ROOT = Path(__file__).parent.parent

_SKELETON = '''
"""Finds large cap sets."""
import numpy as np
import itertools
import funsearch


@funsearch.run
def evaluate(n: int) -> int:
    """Returns the size of an `n`-dimensional cap set."""
    return priority(n)

@funsearch.evolve
def priority(n: int) -> float:
    """Returns the priority with which we want to add `element` to the cap set."""
    return 0.0

'''
_EXPECTED_INITIAL_PROMPT = '''
"""Finds large cap sets."""
import numpy as np
import itertools


def priority_v0(n: int) -> float:
    """Returns the priority with which we want to add `element` to the cap set."""
    return 0.0


def priority_v1(n: int) -> float:
    """Improved version of `priority_v0`."""

'''

_SAMPLE_A = """\
    priority = element
    #######
    # Code from lowest-scoring sampled program.
    #######
    return ...\
"""
_SAMPLE_B = """\
    priority = element ** 2
    #######
    # Code from highest-scoring sampled program.
    #######
    return ...\
"""

_EXPECTED_PROMPT = '''
"""Finds large cap sets."""
import numpy as np
import itertools


def priority_v0(n: int) -> float:
    """Returns the priority with which we want to add `element` to the cap set."""
    priority = element
    #######
    # Code from lowest-scoring sampled program.
    #######
    return ...


def priority_v1(n: int) -> float:
    """Improved version of `priority_v0`."""
    priority = element ** 2
    #######
    # Code from highest-scoring sampled program.
    #######
    return ...


def priority_v2(n: int) -> float:
    """Improved version of `priority_v1`."""

'''


def test_database_integrity():
    backup_data = ROOT / "data" / "backups"
    for file in backup_data.glob("*.pickle"):
        with file.open("rb") as f:
            database = ProgramsDatabase.load(f)
            for island_id, island in enumerate(database._islands):
                last_score = float("-inf")
                for run_id, _function in island._improvements:
                    score = island._runs[run_id]
                    assert score is not None, (
                        f"Islands' improvements should have successful runs (database {file.name}, island {island_id})"
                    )
                    assert score > last_score, (
                        f"Islands' improvements should be strictly increasing (database {file.name}, island {island_id})"
                    )
                    last_score = score


class TestProgramsDatabase:
    def test_initial_prompt(self, tmp_path):
        """Verifies that the first prompt looks as expected."""

        database = ProgramsDatabase(
            config=ProgramsDatabaseConfig(
                inputs=[3],
                specification=_SKELETON,
                problem_name="unused",
                message="unused",
                functions_per_prompt=5,
                num_islands=10,
                reset_period=1000,
                cluster_sampling_temperature_init=1.0,
                cluster_sampling_temperature_period=1000,
            ),
            initial_log_path=tmp_path,
        )

        # Verify the first prompt.
        assert database.get_prompt().code == _EXPECTED_INITIAL_PROMPT

        # Test saving database
        with tempfile.TemporaryFile() as f:
            database.save(f)
            f.seek(0)
            db2 = ProgramsDatabase.load(f)
            assert db2.get_prompt().code == _EXPECTED_INITIAL_PROMPT

    def test_generate_prompt(self):
        """Tests that we build the prompt shown in the paper."""
        template = code_manipulation.text_to_program(_SKELETON)
        function_to_evolve = "priority"
        island = Island(
            template=template,
            function_to_evolve=function_to_evolve,
            functions_per_prompt=2,
            cluster_sampling_temperature_init=1.0,
            cluster_sampling_temperature_period=30_000,
            initial_best_program=None,
            initial_best_scores_per_test={"unused": 1},
        )
        sample_a = copy.deepcopy(template.get_function(function_to_evolve))
        sample_a.body = _SAMPLE_A
        sample_b = copy.deepcopy(template.get_function(function_to_evolve))
        sample_b.body = _SAMPLE_B
        prompt = island._generate_prompt([sample_a, sample_b])
        assert prompt == _EXPECTED_PROMPT

    def test_reset_islands(self, tmp_path):
        template = code_manipulation.text_to_program(_SKELETON)
        function_to_evolve = "priority"
        database = ProgramsDatabase(
            config=ProgramsDatabaseConfig(
                inputs=[3],
                specification=_SKELETON,
                problem_name="unused",
                message="unused",
                functions_per_prompt=5,
                num_islands=10,
                reset_period=1000,
                cluster_sampling_temperature_init=1.0,
                cluster_sampling_temperature_period=1000,
            ),
            initial_log_path=tmp_path,
        )
        scores = [7, 3, 5, 6, 7, -2, 0, -1, 4, 3]
        unused_function = template.get_function(function_to_evolve)
        for i, score in enumerate(scores):
            database.register_program_in_island(
                program=unused_function, island_id=i, scores_per_test={"unused_input": score}
            )

        database.register_program_in_island(unused_function, island_id=7, scores_per_test={"unused_input": 17})
        expected_scores = scores.copy()
        expected_scores[7] = 17
        for i, score in enumerate(expected_scores):
            assert database._islands[i]._best_score == max(score, 0.0), (
                "Score should be at least 0.0 (score of initial program)"
            )

        np.random.seed(0)
        database._reset_islands()
        expected_kept = {0, 2, 3, 4, 7}
        min_kept = min(expected_scores[i] for i in expected_kept)
        for i, score in enumerate(expected_scores):
            if i in expected_kept:
                assert database._islands[i]._best_score == score
            else:
                assert database._islands[i]._best_score >= min_kept

    @pytest.mark.parametrize(
        "logits",
        [
            np.array([10, 9, -1000], dtype=np.float32),
            np.array([10, 9, -1000], dtype=np.int32),
            np.zeros(50),
        ],
    )
    def test_softmax(self, logits: np.ndarray):
        probs_lower_temp = programs_database._softmax(logits, temperature=1.0)
        assert np.isclose(np.sum(probs_lower_temp), 1.0)
        assert np.all(probs_lower_temp >= 0)
        assert np.all(probs_lower_temp <= 1)

        probs_higher_temp = programs_database._softmax(logits, temperature=5.0)
        assert np.isclose(np.sum(probs_higher_temp), 1.0)
        assert np.all(probs_higher_temp >= 0)
        assert np.all(probs_higher_temp <= 1)

        if not np.all(logits == logits[0]):
            # The lower the temperature, the more confident we are on our top choice.
            assert np.max(probs_lower_temp) > np.max(probs_higher_temp)

    @pytest.mark.parametrize(
        ("temperature", "expected"),
        [
            (1, [0.6590012, 0.242433, 0.0985659]),
            (2, [0.50168777, 0.304289, 0.19402324]),
        ],
    )
    def test_softmax_output(self, temperature, expected):
        logits = np.array([2.0, 1.0, 0.1])
        probabilities = programs_database._softmax(logits, temperature)
        np.testing.assert_array_almost_equal(probabilities, np.array(expected), decimal=5)

    @pytest.mark.parametrize(
        "logits",
        [
            np.array([100, 200, 300, np.nan]),
            np.array([100, np.inf, 300, 200]),
            np.array([-np.inf, 200, 300, 100]),
        ],
    )
    def test_softmax_non_finite_error(self, logits):
        with pytest.raises(ValueError, match=r"`logits` contains non-finite value\(s\)"):
            programs_database._softmax(logits, temperature=1.0)


_PY_PROMPT = '''\
import itertools
import jax


@funsearch.run
@jax.jit
def run(n: int):
  return capset(n)


@funsearch.evolve
def capset(n: int):
  """Trivial implementation of capset.

  Args: ...
  """
  return [[1,] * n]
'''

_PY_PROMPT_EVOLVE_RUN = """\
import itertools


@funsearch.run
@funsearch.evolve
def capset(n: int):
  return [[1,] * n]
"""

_PY_PROMPT_NO_RUN = '''\
import itertools


def run(n: int):
  return capset(n)

@funsearch.evolve
def capset(n: int):
  """Trivial implementation of capset.

  Args: ...
  """
  return [[1,] * n]
'''

_PY_PROMPT_NO_EVOLVE = '''\
import itertools


@funsearch.run
def run(n: int):
  return capset(n)


def capset(n: int):
  """Trivial implementation of capset.

  Args: ...
  """
  return [[1,] * n]
'''

_PY_PROMPT_DOUBLE_RUN = '''\
import itertools

@funsearch.run
def run(n: int):
  return capset(n)

@funsearch.run
def capset(n: int):
  """Trivial implementation of capset.

  Args: ...
  """
  return [[1,] * n]
'''


class TestExtractFunctionNames:
    def test_extract_function_names(self):
        to_evolve, to_run = extract_function_names(_PY_PROMPT)
        assert to_run == "run"
        assert to_evolve == "capset"

    def test_extract_function_names_evolve_and_run(self):
        to_evolve, to_run = extract_function_names(_PY_PROMPT_EVOLVE_RUN)
        assert to_run == "capset"
        assert to_evolve == "capset"

    def test_extract_function_names_no_run(self):
        with pytest.raises(ValueError, match=r"Expected 1 function decorated with `@funsearch.run`."):
            extract_function_names(_PY_PROMPT_NO_RUN)

    def test_extract_function_names_no_evolve(self):
        with pytest.raises(ValueError, match=r"Expected 1 function decorated with `@funsearch.evolve`."):
            extract_function_names(_PY_PROMPT_NO_EVOLVE)

    def test_extract_function_names_double_run(self):
        with pytest.raises(ValueError, match=r"Expected 1 function decorated with `@funsearch.run`."):
            extract_function_names(_PY_PROMPT_DOUBLE_RUN)
