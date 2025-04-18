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

"""Class for evaluating programs proposed by the Sampler."""

from __future__ import annotations

import ast
import copy
import re
from typing import Any

from funsearch import code_manipulation, sandbox


class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
        """Collects the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None  # Check internal correctness.
        return self._function_end_line


def _find_method_implementation(generated_code: str, function_to_evolve: str) -> tuple[str, str]:
    """Find the last 'def priority_vX()' method from generated code.

    Return the code and the name of the method.
    """
    """
    Regex to find all methods named 'priority_vX'.
    With each match, start from the 'def priority_vX(' and continue until there's a new line with any of
    - a new 'def'
    - ` or ' or # without indentation
    """
    method_matcher = re.compile(
        rf"def {function_to_evolve}_v\d\(.*?\) -> .*:\n(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+"
    )
    method_name_matcher = re.compile(rf"{function_to_evolve}_v\d+")

    matches = method_matcher.findall(generated_code)
    if not matches:
        return "", ""
    last_match = matches[-1]
    search = method_name_matcher.search(last_match)
    if search is None:
        return "", ""  # This shouldn't happen.
    name = search.group()
    return last_match, name


def _trim_function_body(generated_code: str, function_to_evolve: str) -> str:
    """Extracts the body of the generated function, trimming anything after it."""
    if not generated_code:
        return ""
    if type(generated_code) is not str:
        generated_code = str(generated_code)

    method_name = "fake_function_header"
    # Check is the response only a continuation for our prompt or full method implementation with header
    if f"def {function_to_evolve}_v" in generated_code:
        code, method_name = _find_method_implementation(generated_code, function_to_evolve)
    else:
        code = f"def {method_name}():\n{generated_code}"

    # Finally parse the code to make sure it's valid Python
    tree = None
    # We keep trying and deleting code from the end until the parser succeeds.
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            code = "\n".join(code.splitlines()[: (e.lineno or 0) - 1])
    if not code:
        # Nothing could be saved from `generated_code`
        return ""

    visitor = _FunctionLineVisitor(method_name)
    visitor.visit(tree)
    body_lines = code.splitlines()[1 : visitor.function_end_line]
    return "\n".join(body_lines) + "\n\n"


def _sample_to_program(
    generated_code: str, version_generated: int | None, template: code_manipulation.Program, function_to_evolve: str
) -> tuple[code_manipulation.Function, str]:
    """Returns the compiled generated function and the full runnable program."""
    body = _trim_function_body(generated_code, function_to_evolve)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            body, f"{function_to_evolve}_v{version_generated}", function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    return evolved_function, str(program)


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Returns whether the generated function is calling an earlier version."""
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    return any(name.startswith(f"{function_to_evolve}_v") for name in code_manipulation.get_functions_called(program))


class Evaluator:
    """Class that analyses functions generated by LLMs."""

    def __init__(
        self,
        sandbox: sandbox.ExternalProcessSandbox,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        inputs: list[float] | list[str],
    ) -> None:
        self._sandbox = sandbox
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs

    def analyse(
        self, sample: str, version_generated: int | None, index: int
    ) -> tuple[code_manipulation.Function, dict[float | str, float]]:
        """Compile the sample, execute it on test inputs, returning the compiled function and outputs."""
        match = re.search(r"(```(python|py|))(.*?)```", sample, re.DOTALL)
        code: str = match.group(3) if match else sample

        new_function, program = _sample_to_program(code, version_generated, self._template, self._function_to_evolve)

        scores_per_test: dict[float | str, float] = {}
        for current_input in self._inputs:
            if _calls_ancestor(program, self._function_to_evolve):
                continue
            result = self._sandbox.run(program, self._function_to_run, current_input, index)
            if result is not None:
                scores_per_test[current_input] = result

        return new_function, scores_per_test
