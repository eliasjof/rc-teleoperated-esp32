import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['axes.grid'] = True
rcParams['font.size'] = 18


##### Check functions

def check_inside_circle(point, pos_circle=[0.0, 0.0], r_circle=1):
        
        if (pos_circle[0] - point[0])**2 + (pos_circle[1] - point[1])**2 <=r_circle**2:
            return True
        return False

def check_inside_sphere(point, pos_circle=[0.0, 0.0, 0.0], r_circle=1):
        
        if (pos_circle[0] - point[0])**2 + (pos_circle[1] - point[1])**2 + (pos_circle[2] - point[2])**2 <=r_circle**2:
            return True
        return False


def check_inside_ellipsoid(point, pos_circle=[0.0, 0.0, 0.0], r_circle=[1.,1.,1.]):
        
        if ((pos_circle[0] - point[0])**2)/r_circle[0]**2 + ((pos_circle[1] - point[1])**2)/r_circle[1]**2 + \
           ((pos_circle[2] - point[2])**2)/r_circle[2]**2 <=1:
            return True
        return False

def check_in_line_ab(point, line_a=1, line_b=0):
    if (line_a*point[0] + line_b == point[1]):
        return True
    return False
    
def check_in_line(point, line_p=np.array([0,0]), line_v=np.array([0,1])):
    p = np.array([line_p[1]-point[1],-line_p[0]+point[0]])
    if line_v.dot(p) == 0:
        return True
    return False



#### Random functions


def randomColor(n=1):
    '''Random a color to use in matplotlib
    n = number of colors returned 
    '''
    colors = []
    for i in range(n):
        colors.append("#"+''.join(np.random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']) for i in range(6)))   
    
    if n==1: return colors[0]
    return colors

### Plots

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

def draw_disc(p=np.array([0, 0]), r=1, ax=None, color='blue', fill=True, alpha=0.5):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    circle = plt.Circle((p[0], p[1]),radius=r, alpha=alpha)    
    ax.add_artist(circle)
    return ax