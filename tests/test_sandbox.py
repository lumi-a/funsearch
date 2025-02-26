import pathlib
import tempfile

from funsearch.sandbox import ExternalProcessSandbox

test_prog = """
print("running!")
def x(y):
  print(f"Received {y}")
  return y + 1
   """


def test_external_process_sandbox():
    with tempfile.TemporaryDirectory() as d:
        sandbox = ExternalProcessSandbox(pathlib.Path(d))
        ret = sandbox.run(test_prog, "x", 10, 1)
        assert ret == 11
