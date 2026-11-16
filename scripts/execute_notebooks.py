"""Execute notebooks in the local Python kernel; preserve inline visual evidence."""

import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from scripts.prepare_data import ROOT, dump_json


def main():
    evidence = []
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name='python3')
        # Never use a global Python kernel when reproducing this project.
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, km=manager, timeout=600,
                                resources={'metadata': {'path': str(ROOT)}}, allow_errors=False)
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook, path)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        image_count = sum('image/png' in out.get('data', {}) for cell in code_cells for out in cell.outputs)
        assert code_cells and all(cell.execution_count is not None for cell in code_cells)
        assert image_count >= 2, f'{path.name} has no verified inline figures'
        evidence.append({'notebook': path.name, 'executed_code_cells': len(code_cells),
                         'inline_png_outputs': image_count, 'errors': 0,
                         'kernel_python': sys.executable})
        print(f'Executed {path.name}: {len(code_cells)} code cells, {image_count} inline PNGs', flush=True)
    assert len(evidence) == 2
    dump_json(ROOT / 'results/notebook_execution.json', evidence)


if __name__ == '__main__':
    main()
