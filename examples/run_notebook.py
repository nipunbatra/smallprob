"""Execute the examples using the calling Python environment, not a global kernel."""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
import nbformat
from nbclient import NotebookClient


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', default='notebooks/smallprob_examples.ipynb')
    args=parser.parse_args()
    path=root/args.notebook
    (root/'work').mkdir(exist_ok=True)
    (root/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root/'work',prefix='notebook-kernel-') as temporary:
        folder=Path(temporary)
        kernel=folder/'kernels/smallprob-local'
        kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({
            'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Smallprob current Python','language':'python'}))
        original={key:os.environ.get(key) for key in ['JUPYTER_PATH','JUPYTER_RUNTIME_DIR']}
        os.environ['JUPYTER_PATH']=str(folder)+(os.pathsep+original['JUPYTER_PATH'] if original['JUPYTER_PATH'] else '')
        os.environ['JUPYTER_RUNTIME_DIR']=str(folder/'runtime')
        try:
            notebook=nbformat.read(path,as_version=4)
            NotebookClient(notebook,timeout=240,kernel_name='smallprob-local',
                           resources={'metadata':{'path':str(root)}}).execute()
            notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
            nbformat.validate(notebook)
            nbformat.write(notebook,path)
        finally:
            for key,value in original.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value
    cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    assert all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type=='error' for cell in cells for output in cell.outputs)
    print(f'Executed {len(cells)} code cells with no errors: {path}')


if __name__=='__main__':
    main()
