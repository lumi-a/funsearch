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

"""Tools for manipulating Python code.

It implements 2 classes representing unities of code:
- Function, containing all the information we need about functions: name, args,
  body and optionally a return type and a docstring.
- Program, which contains a code preface (which could be imports, global
  variables and classes, ...) and a list of Functions.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import logging
import re
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableSet, Sequence


@dataclasses.dataclass
class Function:
    """A parsed Python function."""

    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None

    def __str__(self) -> str:
        return_type = f" -> {self.return_type}" if self.return_type else ""

        function = f"def {self.name}({self.args}){return_type}:\n"
        if self.docstring:
            # self.docstring is already indented on every line except the first one.
            # Here, we assume the indentation is always four spaces.
            new_line = "\n" if self.body else ""
            function += f'    """{self.docstring}"""{new_line}'
        # self.body is already indented.
        function += self.body + "\n\n"
        return function

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`.
        if name == "body":
            value = value.strip("\n")
        # Ensure there aren't leading & trailing quotes in `docstring``.
        if name == "docstring" and value is not None and '"""' in value:
            value = value.strip()
            value = value.replace('"""', "")
        super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
    """A parsed Python program."""

    # `preface` is everything from the beginning of the code till the first
    # function is found.
    preface: str
    functions: list[Function]

    def __str__(self) -> str:
        program = f"{self.preface}\n" if self.preface else ""
        program += "\n".join([str(f) for f in self.functions])
        return program

    def find_function_index(self, function_name: str) -> int:
        """Returns the index of input function name."""
        function_names = [f.name for f in self.functions]
        count = function_names.count(function_name)
        if count == 0:
            msg = f"function {function_name} does not exist in program:\n{self!s}"
            raise ValueError(msg)
        if count > 1:
            msg = f"function {function_name} exists more than once in program:\n{self!s}"
            raise ValueError(msg)
        return function_names.index(function_name)

    def get_function(self, function_name: str) -> Function:
        index = self.find_function_index(function_name)
        return self.functions[index]


class ProgramVisitor(ast.NodeVisitor):
    """Parses code to collect all required information to produce a `Program`.

    Note that we do not store function decorators.
    """

    def __init__(self, sourcecode: str) -> None:
        self._codelines: list[str] = sourcecode.splitlines()
        self._preface: str = ""
        self._functions: list[Function] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collects all information about the function being parsed."""
        if node.col_offset == 0:  # Only process top-level functions.
            if not self._functions:
                preface_lines = self._codelines[: node.lineno - 1]
                filtered_lines = []
                in_multiline_import = False

                for line in preface_lines:
                    if in_multiline_import:
                        # Check if the current line contains a closing parenthesis,
                        # which signals the end of a multi-line import.
                        if ")" in line:
                            in_multiline_import = False
                        continue

                    # Check for the start of a multi-line import from funsearch.
                    if re.search(r"^\s*from\s+funsearch\s+import\s*\(", line):
                        # If the closing parenthesis is not on the same line,
                        # mark that we're in a multi-line import block.
                        if not re.search(r"\)", line):
                            in_multiline_import = True
                        continue

                    # Remove single-line funsearch imports.
                    if re.search(r"^\s*(?:from\s+funsearch\s+import\b|import\s+funsearch\b)", line):
                        continue

                    # Remove decorators like @funsearch.run or @funsearch.evolve.
                    if re.search(r"^\s*@funsearch\.(?:run|evolve)\b", line):
                        continue

                    filtered_lines.append(line)

                self._preface = "\n".join(filtered_lines)

            body_start_line: int = node.body[0].lineno - 1
            # Not great to put node.body[-1].lineno as an alternative (e.g. the last statement could be
            # a function-def itself), but I think end_lineno is never None anyway.
            function_end_line: int = node.end_lineno or node.body[-1].lineno + 1
            # Extract the docstring if available.
            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                # TODO: Shouldn't we just extract the inner part, i.e. without the triple quotes?
                # Check what Function.docstring wants to do with it.
                docstring = f'    """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
                body_start_line = node.body[1].lineno - 1 if len(node.body) > 1 else function_end_line

            self._functions.append(
                Function(
                    name=node.name,
                    args=ast.unparse(node.args),
                    return_type=ast.unparse(node.returns) if node.returns else None,
                    docstring=docstring,
                    body="\n".join(self._codelines[body_start_line:function_end_line]),
                )
            )
        self.generic_visit(node)

    def return_program(self) -> Program:
        return Program(preface=self._preface, functions=self._functions)


def text_to_program(text: str) -> Program:
    """Returns Program object by parsing input text using Python AST."""
    try:
        # We assume that the program is composed of some preface (e.g. imports,
        # classes, assignments, ...) followed by a sequence of functions.
        tree = ast.parse(text)
        visitor = ProgramVisitor(text)
        visitor.visit(tree)
        return visitor.return_program()
    except Exception as e:
        logging.warning("Failed parsing %s", text)
        raise e


def text_to_function(text: str) -> Function:
    """Returns Function object by parsing input text using Python AST."""
    program = text_to_program(text)
    if len(program.functions) != 1:
        msg = f"Only one function expected, got {len(program.functions)}:\n{program.functions}"
        raise ValueError(msg)
    return program.functions[0]


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    """Transforms `code` into Python tokens."""
    code_bytes = code.encode()
    code_io = io.BytesIO(code_bytes)
    return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    """Transforms a list of Python tokens into code."""
    code_bytes = tokenize.untokenize(tokens)
    return code_bytes.decode()


def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    """Yields each token with a bool indicating whether it is a function call."""
    try:
        tokens = _tokenize(code)
        prev_token = None
        is_attribute_access = False
        for token in tokens:
            if (
                prev_token  # If the previous token exists and
                and prev_token.type == tokenize.NAME  # it is a Python identifier
                and token.type == tokenize.OP  # and the current token is a delimiter
                and token.string == "("
            ):  # and in particular it is '('.
                yield prev_token, not is_attribute_access
                is_attribute_access = False
            elif prev_token:
                is_attribute_access = prev_token.type == tokenize.OP and prev_token.string == "."
                yield prev_token, False
            prev_token = token
        if prev_token:
            yield prev_token, False
    except Exception as e:
        logging.warning("Failed parsing %s", code)
        raise e


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """Renames function calls from `source_name` to `target_name`."""
    if source_name not in code:
        return code
    modified_tokens = []
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            # Replace the function name token
            modified_token = tokenize.TokenInfo(
                type=token.type, string=target_name, start=token.start, end=token.end, line=token.line
            )
            modified_tokens.append(modified_token)
        else:
            modified_tokens.append(token)
    return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
    """Returns the set of all functions called in `code`."""
    return {token.string for token, is_call in _yield_token_and_is_call(code) if is_call}


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
    """Yields names of functions decorated with `@module.name` in `code`."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and decorator.attr == name:
                    value = decorator.value
                    if isinstance(value, ast.Name) and value.id == module:
                        yield node.name
