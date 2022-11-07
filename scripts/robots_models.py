import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import imageio
rcParams['axes.grid'] = True
rcParams['font.size'] = 18
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
from __utils import *

import csv

class robot_3dholonomic_velocity:
    def __init__(self, r = 1, x0 = 0, y0 = 0, z0 =.0, dt = 0.01, color='red', marker='.', text_id='A'):
        self.r = r
        self.x = [x0]
        self.y = [y0]
        self.z = [z0]        
        self.t = 0.0
        self.dt = dt
        self.color = color
        self.text_id = text_id
        self.marker = marker
        
    def step(self,vx, vy, vz):
        x = self.x[-1] + vx*self.dt
        y = self.y[-1] + vy*self.dt
        z = self.z[-1] + vz*self.dt
        
        self.t += self.dt
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
    
    @staticmethod
    def plot_ellipsoid(center=(0,0,0), dimensions=(0.1,0.1,0.1), ax=None, Npoints=100, color='red', alpha=0.2):

        u = np.linspace(0.0, 2.0 * np.pi, Npoints)
        v = np.linspace(0.0, np.pi, Npoints)
        z = center[2] + dimensions[2]*np.outer(np.cos(u), np.sin(v))
        y = center[1] + dimensions[1]*np.outer(np.sin(u), np.sin(v))
        x = center[0] + dimensions[0]*np.outer(np.ones_like(u), np.cos(v))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=alpha)
        
        return ax
    
    def plot_robot(self, ax, k = None, plot_path=True, kkeep=None): 
        if k is None:
            
            ax.scatter(self.x[-1], self.y[-1], self.z[-1], c= self.color, marker=self.marker)
            ax = self.plot_ellipsoid(center=(self.x[-1], self.y[-1], self.z[-1]), dimensions=(self.r, self.r, self.r), ax=ax, color= self.color)
            # circle = plt.Circle((self.x[-1], self.y[-1]),radius=self.r, c=self.color, fill=False)
            if isinstance(self.x[-1], float):
                ax.text(self.x[-1], self.y[-1], self.z[-1], self.text_id, fontsize='smaller')
            else:
                ax.text(self.x[-1][0], self.y[-1][0], self.z[-1][0], self.text_id, fontsize='smaller')
            if plot_path:
            
                ax.plot3D(self.x, self.y, self.z, marker='.', c=self.color,  alpha=0.4)
            
        elif kkeep is None:
            
            ax.scatter(self.x[k], self.y[k], self.z[k], c= self.color, marker=self.marker)
            ax = self.plot_ellipsoid(center=(self.x[k], self.y[k], self.z[k]), dimensions=(self.r, self.r, self.r), ax=ax, color= self.color)
            # circle = plt.Circle((self.x[k], self.y[k]),radius=self.r, c=self.color, fill=False)
            # print(self.x[k], self.y[k],  self.z[k])
            if not isinstance(self.x[k], float):
                ax.text(self.x[k][0], self.y[k][0],  self.z[k][0], self.text_id,fontsize='smaller')
            else:
                ax.text(self.x[k], self.y[k],  self.z[k], self.text_id,fontsize='smaller')
            if plot_path:
                ax.plot3D(self.x[:k+1], self.y[:k+1],  self.z[:k+1], marker='.', c=self.color,  alpha=0.4)
        
        else:
            ax.scatter(self.x[k], self.y[k], self.z[k], c= self.color, marker=self.marker)
            ax = self.plot_ellipsoid(center=(self.x[k], self.y[k], self.z[k]), dimensions=(self.r, self.r, self.r), ax=ax, color= self.color)
            # circle = plt.Circle((self.x[k], self.y[k]),radius=self.r, c=self.color, fill=False)
            # print(self.x[k], self.y[k],  self.z[k])
            if not isinstance(self.x[k], float):
                ax.text(self.x[k][0], self.y[k][0],  self.z[k][0], self.text_id,fontsize='smaller')
            else:
                ax.text(self.x[k], self.y[k],  self.z[k], self.text_id,fontsize='smaller')
            if plot_path:
                if k - kkeep <= 0:
                    ax.plot3D(self.x[:k+1], self.y[:k+1],  self.z[:k+1], marker='.', c=self.color,  alpha=0.4)
                else:
                    ax.plot3D(self.x[kkeep:k+1], self.y[kkeep:k+1],  self.z[kkeep:k+1], marker='.', c=self.color,  alpha=0.4)
        # ax.add_artist(circle)
        return ax

    def savepath(self, file='path.csv'):
        with open(file, 'w', newline='') as student_file:
            writer = csv.writer(student_file)
            writer.writerow(self.x)
            writer.writerow(self.y)
            writer.writerow(self.z)
            

class robot_3dholonomic_acceleration:
    def __init__(self, r = 1, x0 = 0, y0 = 0, z0 =.0, dt = 0.01, color='red', marker='.', text_id='A'):
        self.r = r
        self.x = [x0]
        self.y = [y0]
        self.z = [z0]
        self.vx = [0.]
        self.vy = [0.]
        self.vz = [0.]
        self.ax = [0.]
        self.ay = [0.]
        self.az = [0.]
        self.xcurrent = np.array([self.x[-1], self.y[-1], self.z[-1]])
        self.vcurrent = np.array([self.vx[-1], self.vy[-1], self.vz[-1]])
        self.t = 0.0
        self.dt = dt
        self.color = color
        self.text_id = text_id
        self.marker = marker
        
    def step(self,ax, ay, az):
        x = self.x[-1] + self.vx[-1]*self.dt  + 0.5*ax*(self.dt**2)
        y = self.y[-1] + self.vy[-1]*self.dt  + 0.5*ay*(self.dt**2)
        z = self.z[-1] + self.vz[-1]*self.dt  + 0.5*az*(self.dt**2)
        vx = self.vx[-1] + ax*self.dt
        vy = self.vy[-1] + ay*self.dt
        vz = self.vz[-1] + az*self.dt
        
        
        
        self.t += self.dt
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.vx.append(vx)
        self.vy.append(vy)
        self.vz.append(vz)
        self.ax.append(ax)
        self.ay.append(ay)
        self.az.append(az)
        self.xcurrent = np.array([self.x[-1], self.y[-1], self.z[-1]])
        self.vcurrent = np.array([self.vx[-1], self.vy[-1], self.vz[-1]])
    
    @staticmethod
    def plot_ellipsoid(center=(0,0,0), dimensions=(0.1,0.1,0.1), ax=None, Npoints=100, color='red', alpha=0.2):

        u = np.linspace(0.0, 2.0 * np.pi, Npoints)
        v = np.linspace(0.0, np.pi, Npoints)
        z = center[2] + dimensions[2]*np.outer(np.cos(u), np.sin(v))
        y = center[1] + dimensions[1]*np.outer(np.sin(u), np.sin(v))
        x = center[0] + dimensions[0]*np.outer(np.ones_like(u), np.cos(v))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=alpha)
        
        return ax
    
    def plot_robot(self, ax, k = None, plot_path=True, kkeep=None, alpha=0.4, fontsize='smaller'): 
        if k is None:
            
            ax.scatter(self.x[-1], self.y[-1], self.z[-1], c= self.color, marker=self.marker)
            ax = self.plot_ellipsoid(center=(self.x[-1], self.y[-1], self.z[-1]), dimensions=(self.r, self.r, self.r), ax=ax, color= self.color)
            # circle = plt.Circle((self.x[-1], self.y[-1]),radius=self.r, c=self.color, fill=False)
            if isinstance(self.x[-1], float):
                ax.text(self.x[-1], self.y[-1], self.z[-1], self.text_id, fontsize=fontsize)
            else:
                ax.text(self.x[-1][0], self.y[-1][0], self.z[-1][0], self.text_id, fontsize=fontsize)
            if plot_path:
            
                ax.plot3D(self.x, self.y, self.z, marker='.', c=self.color,  alpha=alpha)
            
        elif kkeep is None:
            
            ax.scatter(self.x[k], self.y[k], self.z[k], c= self.color, marker=self.marker)
            ax = self.plot_ellipsoid(center=(self.x[k], self.y[k], self.z[k]), dimensions=(self.r, self.r, self.r), ax=ax, color= self.color)
            # circle = plt.Circle((self.x[k], self.y[k]),radius=self.r, c=self.color, fill=False)
            # print(self.x[k], self.y[k],  self.z[k])
            if not isinstance(self.x[k], float):
                ax.text(self.x[k][0], self.y[k][0],  self.z[k][0], self.text_id,fontsize=fontsize)
            else:
                ax.text(self.x[k], self.y[k],  self.z[k], self.text_id,fontsize=fontsize)
            if plot_path:
                ax.plot3D(self.x[:k+1], self.y[:k+1],  self.z[:k+1], marker='.', c=self.color,  alpha=alpha)
        
        else:
            
            ax.scatter(self.x[k], self.y[k], self.z[k], c= self.color, marker=self.marker)
            ax = self.plot_ellipsoid(center=(self.x[k], self.y[k], self.z[k]), dimensions=(self.r, self.r, self.r), ax=ax, color= self.color)
            # circle = plt.Circle((self.x[k], self.y[k]),radius=self.r, c=self.color, fill=False)
            # print(self.x[k], self.y[k],  self.z[k])
            if not isinstance(self.x[k], float):
                ax.text(self.x[k][0], self.y[k][0],  self.z[k][0], self.text_id,fontsize=fontsize)
            else:
                ax.text(self.x[k], self.y[k],  self.z[k], self.text_id,fontsize=fontsize)
            if plot_path:
                if k - kkeep < 0:
                    ax.plot3D(self.x[:k+1], self.y[:k+1],  self.z[:k+1], marker='.', c=self.color,  alpha=0.4)
                else:
                    ax.plot3D(self.x[k - kkeep:k+1], self.y[k - kkeep:k+1],  self.z[k - kkeep:k+1], marker='.', c=self.color,  alpha=0.4)
        
        # ax.add_artist(circle)
        return ax
    def savepath(self, file='path.csv'):
        with open(file, 'w', newline='') as student_file:
            writer = csv.writer(student_file)
            writer.writerow(self.x)
            writer.writerow(self.y)
            writer.writerow(self.z)
            
