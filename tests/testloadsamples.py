import crbm
import numpy as np

def test_load_sample():
    x = crbm.load_sample()
    np.testing.assert_equal(x.shape, (3997, 1, 4, 200))
