"""One command: create a local environment, install, run and verify everything."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import venv

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    local_python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        if not local_python.exists():
            print('Creating project-local .venv', flush=True)
            venv.create(ROOT / '.venv', with_pip=True)
        digest = hashlib.sha256((ROOT / 'requirements.txt').read_bytes()).hexdigest()
        stamp = ROOT / '.venv' / '.requirements-sha256'
        if not stamp.exists() or stamp.read_text() != digest:
            install_env = dict(os.environ, PIP_CACHE_DIR=str(ROOT / '.venv' / 'pip-cache'))
            subprocess.run([str(local_python), '-m', 'pip', 'install', '-r', 'requirements.txt'],
                           env=install_env, check=True)
            stamp.write_text(digest)
        subprocess.run([str(local_python), '-u', str(Path(__file__).resolve())], check=True)
        return

    (ROOT / 'results').mkdir(exist_ok=True)
    started = time.perf_counter()
    env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', IPYTHONDIR=str(ROOT / '.venv' / 'ipython'),
               JUPYTER_DATA_DIR=str(ROOT / '.venv' / 'jupyter'),
               JUPYTER_RUNTIME_DIR=str(ROOT / '.venv' / 'jupyter' / 'runtime'),
               MPLCONFIGDIR=str(ROOT / '.venv' / 'matplotlib'))
    steps = ['scripts.prepare_data', 'scripts.train_model', 'scripts.predict',
             'scripts.execute_notebooks', 'scripts.verify_project']
    with (ROOT / 'results' / 'execution.log').open('w', encoding='utf-8') as log:
        for module in steps:
            line = f'\nRUN {sys.executable} -m {module}\n'
            print(line, end='', flush=True)
            log.write(line)
            process = subprocess.Popen([sys.executable, '-u', '-m', module], cwd=ROOT, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f'{module} failed; see results/execution.log')
        line = f'\nSUCCESS: complete pipeline verified in {time.perf_counter() - started:.1f}s\n'
        print(line, end='')
        log.write(line)
    with (ROOT / 'results' / 'requirements-lock.txt').open('w') as handle:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)


if __name__ == '__main__':
    main()
