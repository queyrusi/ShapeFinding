import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from EllipseGenerator import *

__name__ = 'main'


if __name__ == 'main':
    ellipse_gen = Ellipse()
    ellipse_gen.run(semimaj=5,semimin=1,phi=np.pi/3,
     theta_num=50, x_cent=3, y_cent=-2)
    print("Data ellipse ", ellipse_gen.data)
    # Execute success

    # Must include unit-test
    # Ellipse should be produced by n_points (density), arc, etc.
	# Test 1 success