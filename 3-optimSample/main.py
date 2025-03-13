import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def main():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau(np.array([X, Y]))
    
    fig=plt.figure("Himmelblau")
    # ax=fig.gca(projection='3d')
    ax = fig.add_axes(Axes3D(fig)) 
    ax.plot_surface(X, Y, Z)
    ax.view_init(60,-30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    

    # [1.,0.], [-4.,0.], [4.,0.]
    x = torch.tensor([0.,0.], requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=1e-3)
    for step in range(20000):
        pred = himmelblau(x)
        optimizer.zero_grad()
        pred.backward()
        optimizer.step()
        
        if step % 2000 == 0:
            print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))
            
if __name__ == '__main__':
    main()