import matplotlib
matplotlib.use('Agg')

import os
from crbm.tutorial import tutorial

def test_tutorial(tmpdir):

    path = tmpdir.mkdir('run')

    tutorial(path.strpath)

    outputs = [ o.basename for o in path.listdir() ]
    #print([ o.basename for o in outputs])

    assert 'oct4_model_params.pkl' in outputs
    assert 'pfms' in outputs
    assert 'logos' in outputs
    assert 'logo1.png' in outputs
    assert 'densityplot.png' in outputs
    assert 'tsnescatter.png' in outputs
    assert 'tsnescatter_pies.png' in outputs
    assert 'violinplot.png' in outputs
    
    for subdir in ['pfms', 'logos']:
        assert len(os.listdir(os.path.join(path.strpath, subdir))) == 10

