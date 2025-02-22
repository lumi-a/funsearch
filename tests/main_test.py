import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from funsearch.__main__ import _parse_input, start, resume, ls

ROOT_DIR = Path(__file__).parent.parent

runner = CliRunner()


class TestMain(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.default_args = [
      "--output-path",
      self.temp_dir,
      "--iterations",
      "1",
      str(ROOT_DIR / "specs" / "cap-set.py"),
      "4",
    ]

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_main(self):
    result = runner.invoke(start, [])
    assert result.exit_code == 2
    assert "Usage:" in result.output
    with patch("funsearch.core.run", return_value=None) as mock_run:
      result = runner.invoke(start, self.default_args)
      assert result.exit_code == 0
      assert mock_run.call_count == 1

  def test_main_sample(self):
    with patch("funsearch.sampler.LLM._draw_sample", return_value="return 0.5") as mock_run:
      result = runner.invoke(start, self.default_args)
      assert result.exit_code == 0


def test_parse_input():
  assert _parse_input("1") == [1]
  assert _parse_input("1,2,3") == [1, 2, 3]
  assert _parse_input(str(ROOT_DIR / "tests" / "fixtures" / "inputs-numeric.json")) == [9, 10, 11]
  assert _parse_input(str(ROOT_DIR / "tests" / "fixtures" / "inputs-string.json")) == ["a", "bc", "def"]
