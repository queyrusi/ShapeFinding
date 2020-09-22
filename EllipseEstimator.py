import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class EllipseEstimator:
    """
    """

    def __init__(self, data):
        """
        """
        self.data = None
        return


    def estimate(self, method='Powell'):
        # semimaj=5,semimin=1,phi=np.pi/3, theta_num=50, x_cent=3, y_cent=-2

        # A priori ; theta_num = 50
        candidate_0 = [5., 1., np.pi/3, 3, 2]


if __name__ == 'main':
    plot_ellipse(semimaj=5,semimin=1,phi=np.pi/3, theta_num=50, x_cent=3, y_cent=-2)
    plt.show()


