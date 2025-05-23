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

import itertools
import textwrap

import pytest

from funsearch import code_manipulation

_IMPORTS: str = """\
import itertools
import numpy


"""

_CLASS: str = """\
class Helper:
    def __init__( n: int):
        self.n = n
        self.initial_capset = get_capset_v0(n)


"""

_ASSIGNMENT: str = """\
some_global_variable = 0


"""

_FUNCTIONS: str = '''\
def get_capset_v0(n: int) -> set[tuple[int, ...]]:
    """Computes a cap set for n number of copies.

    A cap set is a subset of an n-dimensional affine space over a three-element
    field with no three elements in a line.

    Args:
        n: an integer, number of copies.

    Returns:
        A set of tuples in {0, 1, 2}.
    """
    capset = set()
    for i in range(n):
        capset.add((0) * i + (1) + (0) * (n - i - 1))
    return capset


def get_capset_v2(k: int):
    """One line docstring."""
    get_capset_v0 = get_capset_v0(k)
    return get_capset_v0\\

'''


_SMALL_PROGRAM: str = """\
def test() -> np.ndarray:
    return np.zeros(1)
"""

_FUNCTION_HEADER: str = "def get_capset_v0(n: int)"
_FUNCTION_RETURN_TYPE: str = " -> set[tuple[int, ...]]"
_FUNCTION_DOCSTRING: str = '    """One line docstring."""'
_FUNCTION_BODY: str = """\
    capset = set()
    for i in range(n):
        capset.add((0) * i + (1) + (0) * (n - i - 1))
    return capset
"""


def create_test_function(has_return_type: bool, has_docstring: bool) -> str:
    code = _FUNCTION_HEADER
    if has_return_type:
        code += _FUNCTION_RETURN_TYPE
    code += ":\n"
    if has_docstring:
        code += _FUNCTION_DOCSTRING
    code += "\n"
    code += _FUNCTION_BODY
    return code


def create_test_program(has_imports: bool, has_class: bool, has_assignment: bool) -> str:
    code = ""
    if has_imports:
        code += _IMPORTS
    if has_class:
        code += _CLASS
    if has_assignment:
        code += _ASSIGNMENT
    code += _FUNCTIONS
    return code


@pytest.mark.parametrize(("has_return_type", "has_docstring"), list(itertools.product([False, True], repeat=2)))
def test_text_to_function(has_return_type: bool, has_docstring: bool):
    function = code_manipulation.text_to_function(create_test_function(has_return_type, has_docstring))
    assert function.name == "get_capset_v0"
    assert function.args == "n: int"
    if has_return_type:
        assert function.return_type == "set[tuple[int, ...]]"
    else:
        assert function.return_type is None
    if has_docstring:
        assert function.docstring == "One line docstring."
    else:
        assert function.docstring is None
    assert function.body == _FUNCTION_BODY.rstrip()


def test_small_text_to_program():
    program = code_manipulation.text_to_program(_SMALL_PROGRAM)
    assert program.preface == ""
    assert len(program.functions) == 1

    expected_function = code_manipulation.Function(
        name="test", args="", return_type="np.ndarray", body="    return np.zeros(1)"
    )
    assert expected_function == program.functions[0]
    assert str(program) == _SMALL_PROGRAM + "\n"

    # Assert that we do not add one more '\n' each time we convert to program.
    program_again = code_manipulation.text_to_program(str(program))
    assert str(program) == str(program_again)


@pytest.mark.parametrize(("has_imports", "has_class", "has_assignment"), [(True, False, False)])
def test_text_to_program(has_imports: bool, has_class: bool, has_assignment: bool):
    code = create_test_program(has_imports, has_class, has_assignment)
    program = code_manipulation.text_to_program(code)
    assert len(program.functions) == 2

    doc = textwrap.dedent(
        """\
    Computes a cap set for n number of copies.

        A cap set is a subset of an n-dimensional affine space over a three-element
        field with no three elements in a line.

        Args:
            n: an integer, number of copies.

        Returns:
            A set of tuples in {0, 1, 2}.
      """
    )
    body = textwrap.dedent(
        """\
      capset = set()
      for i in range(n):
          capset.add((0) * i + (1) + (0) * (n - i - 1))
      return capset"""
    )
    expected_function_0 = code_manipulation.Function(
        name="get_capset_v0",
        args="n: int",
        return_type="set[tuple[int, ...]]",
        docstring=doc + "    ",
        body=textwrap.indent(body, "    "),
    )
    expected_function_1 = code_manipulation.Function(
        name="get_capset_v2",
        args="k: int",
        body="""    get_capset_v0 = get_capset_v0(k)
    return get_capset_v0\\\n\n""",
        docstring="One line docstring.",
    )
    if not has_imports and not has_class and not has_assignment:
        assert program.preface == ""
    if has_imports:
        assert _IMPORTS.rstrip() in program.preface
    if has_class:
        assert _CLASS.strip() in program.preface
    if has_assignment:
        assert _ASSIGNMENT.strip() in program.preface
    assert expected_function_0 == program.functions[0]
    assert expected_function_1 == program.functions[1]
    assert code == str(program)

    # Make sure that one can convert Function to string and then back to a
    # function so that it remains the same.
    for i in range(2):
        assert program.functions[i] == code_manipulation.text_to_function(str(program.functions[i]))


def test_get_functions_called():
    code = textwrap.dedent("""\
        def f(n: int) -> int:
            if n == 1:
                return a(n)
            elif n == 2:
                return b(n) + object.c(n - 1)
            a = object.property
            g()
            return f(n - 1)
      """)
    assert code_manipulation.get_functions_called(code) == {"a", "b", "f", "g"}
