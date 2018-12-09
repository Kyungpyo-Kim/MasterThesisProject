# Example usage of align() method:
# generate two random sets of points as 2 actors 
# and align them using vtkIterativeClosestPointTransform.
# Retrieve the vtk transformation matrix.
#
from __future__ import division, print_function
from random import uniform as u
import plotter

vp = plotter.vtkPlotter(shape=[1,2], verbose=0, axes=2)

N1 = 15  # number of points of first set
N2 = 10  # number of points of second set
x = 1.   # add some randomness

pts1 = [ (u(0,x), u(0,x), u(0,x)+i) for i in range(N1) ]
pts2 = [ (u(0,x)+3, u(0,x)+i/2+2, u(0,x)+i+1) for i in range(N2) ]

act1 = vp.points(pts1, c='b', legend='source')
act2 = vp.points(pts2, c='r', legend='target')

vp.show(at=0, interactive=0)

# find best alignment between the 2 sets of points
alpts1 = vp.align(act1,act2).coordinates()

for i in range(N1): #draw arrows to see where points end up
    vp.arrow(pts1[i], alpts1[i], c='k', s=0.007, alpha=.1) 

vp.show(at=1, interactive=1)


