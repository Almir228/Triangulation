"""Интеграционные проверки: python3 tests/check_cli.py /path/to/triangulation."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
formula = Path(__file__).parents[1]/'tools/formula.py'
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    def run(args, code=0):
        p = subprocess.run([binary, *args], cwd=root, capture_output=True, text=True)
        assert p.returncode == code, (args, p.returncode, p.stdout, p.stderr)
        return p
    run(['--help'])
    run(['--refine', 'abc'], 1)
    run(['--normal', '0', '0', '0'], 1)
    run(['--contour', 'missing.csv'], 1)
    p=root/'bad.csv';p.write_text('0,0,0\n1,0,0\n0,1,0 trailing\n');run(['--contour', str(p)],1)
    # A consistently oriented tetrahedron has no open boundary.
    p=root/'closed.obj';p.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n');run(['--mesh',str(p)],1)
    # Negative OBJ indices and v/vt/vn syntax.
    p=root/'triangle.obj';p.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3/1/1 -2/2/1 -1/3/1\n');run(['--mesh',str(p),'--refine','0'])
    # Both formula entry paths run end to end.
    subprocess.run([sys.executable,str(formula),'contour','--samples','24','--output',str(root/'curve.csv')],check=True,capture_output=True)
    run(['--contour',str(root/'curve.csv'),'--refine','2'])
    rows=list(csv.DictReader((root/'surface.csv').open()))
    assert all(float(b['area']) <= float(a['area'])+1e-11 for a,b in zip(rows, rows[1:]))
    subprocess.run([sys.executable,str(formula),'surface','--z','0.4*(1-x*x)*(1-y*y)','--steps','8','--output',str(root/'dome.obj')],check=True,capture_output=True)
    run(['--mesh',str(root/'dome.obj'),'--normal','0','0','1','--refine','0','--iterations','1'],2)
    run(['--mesh',str(root/'dome.obj'),'--normal','0','0','1','--refine','0'])
    rows=list(csv.DictReader((root/'surface.csv').open()));assert abs(float(rows[-1]['area'])-4)<1e-10
    html = (root/'surface.html').read_text()
    assert '<canvas' in html
    assert 'Скачать WebM' in html and 'MediaRecorder' in html
    assert 'Number(timeline.value)' in html and 'playGeneration' in html
    assert html.count('{v:') >= 2
print('CLI: validation, formula inputs, OBJ indices, monotonic area and nonconvergence status passed')
