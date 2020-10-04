import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Ellipse:
    """This class defines the Ellipse class giving all variations of shapes for parametric curve
    """

    def __init__(self):
        """
        """
        self.data = None
        return

    def generate_observation(self, length, density, section=True):
        # ellipse_param = np.array([x, y, h, k, b, a])

        # Simple tuning of theta_num
        if not section:
            pass
        return

    def plot_ellipse(self,semimaj=1,semimin=1,phi=0,x_cent=0,y_cent=0,theta_num=1e3,ax=None,plot_kwargs=None,\
                        fill=False,fill_kwargs=None,data_out=False,cov=None,mass_level=0.68, plot=False):

        # Get Ellipse Properties from cov matrix
        if cov is not None:
            eig_vec,eig_val,u = np.linalg.svd(cov)
            # Make sure 0th eigenvector has positive x-coordinate
            if eig_vec[0][0] < 0:
                eig_vec[0] *= -1
            semimaj = np.sqrt(eig_val[0])
            semimin = np.sqrt(eig_val[1])
            if mass_level is None:
                multiplier = np.sqrt(2.279)
            else:
                distances = np.linspace(0,20,20001)
                chi2_cdf = chi2.cdf(distances,df=2)
                multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
            semimaj *= multiplier
            semimin *= multiplier
            phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
            if eig_vec[0][1] < 0 and phi > 0:
                phi *= -1

        # Generate data for ellipse structure
        theta = np.linspace(0,2*np.pi,int(theta_num))
        r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        data = np.array([x,y])
        S = np.array([[semimaj,0],[0,semimin]])
        R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        T = np.dot(R,S)
        data = np.dot(T,data)
        data[0] += x_cent
        data[1] += y_cent

        # Plot!
        return_fig = False
        if ax is None:
            return_fig = True
            fig,ax = plt.subplots()

        if plot == True:
            if plot_kwargs is None:
                ax.scatter(data[0],data[1],color='b',linestyle='-')
            else:
                ax.scatter(data[0],data[1],**plot_kwargs)

            if fill == True:
                ax.fill(data[0],data[1],**fill_kwargs)

            if return_fig == True:
                return fig

    def run(self,semimaj=1,semimin=1,phi=0,x_cent=0,y_cent=0,
    	theta_num=1e3,ax=None,plot_kwargs=None,
        fill=False,fill_kwargs=None,cov=None,
        mass_level=0.68, plot=False):
        # Get Ellipse Properties from cov matrix
        if cov is not None:
            eig_vec,eig_val,u = np.linalg.svd(cov)
            # Make sure 0th eigenvector has positive x-coordinate
            if eig_vec[0][0] < 0:
                eig_vec[0] *= -1
            semimaj = np.sqrt(eig_val[0])
            semimin = np.sqrt(eig_val[1])
            if mass_level is None:
                multiplier = np.sqrt(2.279)
            else:
                distances = np.linspace(0,20,20001)
                chi2_cdf = chi2.cdf(distances,df=2)
                multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
            semimaj *= multiplier
            semimin *= multiplier
            phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
            if eig_vec[0][1] < 0 and phi > 0:
                phi *= -1

        # Generate data for ellipse structure
        theta = np.linspace(0,2*np.pi,int(theta_num))
        r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        self.data = np.array([x,y])
        S = np.array([[semimaj,0],[0,semimin]])
        R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        T = np.dot(R,S)
        self.data = np.dot(T,self.data)
        self.data[0] += x_cent
        self.data[1] += y_cent
        return

    def reset(self):
    	self.data = None   	

if __name__ == 'main':
    plot_ellipse(semimaj=5,semimin=1,phi=np.pi/3, theta_num=50, x_cent=3, y_cent=-2)
    plt.show()


