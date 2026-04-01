#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def summarize_source(source: str, max_len: int = 100) -> str:
    for line in source.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= max_len else line[: max_len - 3] + '...'
    return '<empty>'


def find_section_titles(nb):
    current_title = ''
    mapping = {}
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            for line in cell.source.splitlines():
                line = line.strip()
                if line.startswith('#'):
                    current_title = line.lstrip('#').strip()
                    break
        mapping[idx] = current_title
    return mapping


def atomic_write_notebook(nb, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    tmp_path.replace(path)


class LiveNotebookClient(NotebookClient):
    def __init__(self, nb, output_path: Path, **kwargs):
        super().__init__(nb, **kwargs)
        self.output_path = output_path
        self.total_cells = len(nb.cells)
        self.section_titles = find_section_titles(nb)
        self.cell_start_times = {}
        self.on_notebook_start = self._on_notebook_start
        self.on_cell_start = self._on_cell_start
        self.on_cell_executed = self._on_cell_executed
        self.on_cell_error = self._on_cell_error
        self.on_notebook_complete = self._on_notebook_complete
        self.on_notebook_error = self._on_notebook_error

    def emit(self, text: str = '', end: str = '\n') -> None:
        sys.stdout.write(text + end)
        sys.stdout.flush()

    def cell_label(self, cell, cell_index: int) -> str:
        section = self.section_titles.get(cell_index, '')
        snippet = summarize_source(cell.source)
        if section:
            return f'{section} | {snippet}'
        return snippet

    def should_trace_cell(self, cell) -> bool:
        return cell.cell_type == 'code' and bool(cell.source.strip())

    def _on_notebook_start(self, **kwargs):
        code_cells = sum(1 for cell in self.nb.cells if self.should_trace_cell(cell))
        self.emit(f'===== NOTEBOOK START total_cells={self.total_cells} code_cells={code_cells} =====')

    def _on_cell_start(self, cell=None, cell_index=None, **kwargs):
        if cell is None or cell_index is None or not self.should_trace_cell(cell):
            return
        self.cell_start_times[cell_index] = time.time()
        self.emit(f'\n===== CELL {cell_index + 1}/{self.total_cells} START =====')
        self.emit(self.cell_label(cell, cell_index))

    def _on_cell_executed(self, cell=None, cell_index=None, execute_reply=None, **kwargs):
        if cell is None or cell_index is None or not self.should_trace_cell(cell):
            return
        elapsed = time.time() - self.cell_start_times.get(cell_index, time.time())
        status = 'unknown'
        if execute_reply is not None:
            status = execute_reply.get('content', {}).get('status', 'unknown')
        atomic_write_notebook(self.nb, self.output_path)
        self.emit(f'===== CELL {cell_index + 1}/{self.total_cells} END status={status} elapsed={elapsed:.1f}s =====')

    def _on_cell_error(self, cell=None, cell_index=None, execute_reply=None, **kwargs):
        if cell is None or cell_index is None:
            return
        atomic_write_notebook(self.nb, self.output_path)
        status = 'error'
        if execute_reply is not None:
            status = execute_reply.get('content', {}).get('status', 'error')
        self.emit(f'===== CELL {cell_index + 1}/{self.total_cells} ERROR status={status} =====')

    def _on_notebook_complete(self, **kwargs):
        atomic_write_notebook(self.nb, self.output_path)
        self.emit(f'===== NOTEBOOK COMPLETE output={self.output_path} =====')

    def _on_notebook_error(self, **kwargs):
        atomic_write_notebook(self.nb, self.output_path)
        self.emit(f'===== NOTEBOOK ERROR output={self.output_path} =====')

    def output(self, outs, msg, display_id, cell_index):
        out = super().output(outs, msg, display_id, cell_index)
        if out is None:
            return None

        output_type = out.get('output_type')
        if output_type == 'stream':
            text = out.get('text', '')
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
        elif output_type in {'execute_result', 'display_data'}:
            data = out.get('data', {})
            text = data.get('text/plain')
            if text:
                if not text.endswith('\n'):
                    text += '\n'
                sys.stdout.write(text)
                sys.stdout.flush()
            else:
                keys = ', '.join(sorted(data.keys())) or '<no data>'
                self.emit(f'[cell {cell_index + 1}] {output_type}: {keys}')
        elif output_type == 'error':
            traceback_text = '\n'.join(out.get('traceback', []))
            if traceback_text:
                if not traceback_text.endswith('\n'):
                    traceback_text += '\n'
                sys.stdout.write(traceback_text)
                sys.stdout.flush()
        return out


def parse_args():
    parser = argparse.ArgumentParser(description='Execute a notebook with live streamed outputs.')
    parser.add_argument('notebook', help='Path to the input notebook')
    parser.add_argument('--output-dir', required=True, help='Directory for the executed notebook')
    parser.add_argument('--output', required=True, help='Output notebook filename')
    parser.add_argument('--cwd', default=None, help='Working directory passed to the kernel')
    parser.add_argument('--timeout', type=int, default=-1, help='Cell timeout in seconds; -1 disables timeout')
    parser.add_argument('--kernel-name', default=None, help='Override notebook kernel name')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook_path = Path(args.notebook).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_path = output_dir / args.output
    cwd = str(Path(args.cwd).resolve()) if args.cwd else str(notebook_path.parent)

    with notebook_path.open('r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    if args.kernel_name:
        nb.metadata.setdefault('kernelspec', {})['name'] = args.kernel_name

    timeout = None if args.timeout is None or args.timeout < 0 else args.timeout
    client_kwargs = {
        'timeout': timeout,
        'record_timing': True,
    }
    if args.kernel_name:
        client_kwargs['kernel_name'] = args.kernel_name

    client = LiveNotebookClient(
        nb,
        output_path=output_path,
        **client_kwargs,
    )

    try:
        client.execute(cwd=cwd)
    except CellExecutionError:
        atomic_write_notebook(nb, output_path)
        return 1
    except KeyboardInterrupt:
        atomic_write_notebook(nb, output_path)
        print('\n===== NOTEBOOK INTERRUPTED =====', flush=True)
        return 130
    except Exception:
        atomic_write_notebook(nb, output_path)
        raise

    atomic_write_notebook(nb, output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
