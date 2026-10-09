"""Generate the percent-cell notebook and execute it with a new kernel each time."""
import ast
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec


class CurrentPythonKernel(KernelManager):
    """Use this interpreter without installing a global/user kernelspec."""
    @property
    def kernel_spec(self):
        return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                          display_name="Python 3", language="python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-reproducible", action="store_true",
                        help="Execute twice in fresh kernels and compare generated artifact hashes")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    source = directory / "03_modelling_and_comparison.py"
    notebook = directory / "03_modelling_and_comparison.ipynb"
    cells = []
    for part in source.read_text(encoding="utf-8").split("# %%")[1:]:
        header, body = part.split("\n", 1)
        if "[markdown]" in header:
            cells.append(nbformat.v4.new_markdown_cell(ast.literal_eval(body.strip()).strip()))
        else:
            cells.append(nbformat.v4.new_code_cell(body.strip()))
    for i, cell in enumerate(cells):
        cell.id = f"member3-{i:02d}"
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    })
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[variable] = "1"
    previous = None
    for run in range(2 if args.verify_reproducible else 1):
        executed = copy.deepcopy(nb)
        print(f"Fresh-kernel run {run + 1}", flush=True)
        client = NotebookClient(executed, timeout=1200, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(directory)}},
                                kernel_manager_class=CurrentPythonKernel)
        client.on_cell_start = lambda cell, cell_index, **kwargs: print(
            f"Cell {cell_index + 1}/{len(cells)}: {cell.cell_type}", flush=True)
        client.execute()
        nbformat.validate(executed)
        nbformat.write(executed, notebook)
        print(f"Executed {len(cells)} cells successfully: {notebook.name}")
        manifest = directory.parent / "results/run_manifest.json"
        current = json.loads(manifest.read_text(encoding="utf-8"))
        if previous is not None:
            if current != previous:
                raise AssertionError("Fresh-kernel manifests differ; review output reproducibility")
            # Validate actual bytes as well as hashes written by the notebook.
            root = directory.parents[2]
            for relative, expected in current["output_sha256"].items():
                assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
            assert hashlib.sha256((root / "data/processed/cluster_assignments.csv").read_bytes()).hexdigest() == current["assignment_sha256"]
            print("PASS: assignments, result tables, figures, documents and manifests reproduce byte-for-byte.")
        previous = current


if __name__ == "__main__":
    main()
