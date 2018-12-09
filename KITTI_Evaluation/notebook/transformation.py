import numpy as np

""" Transformation """
def transform_pointcloud_affine(pc, mat):
    """
    pc : Nx3 numpy matrix
    mat: 4x4 numpy matrix
    """
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    func_tf = lambda x: unpad(np.dot(mat, pad(x).T).T)
    
    return func_tf(pc)
    
def make_matrix_rpy(r=0,p=0,y=0):
    
    yawMatrix = np.matrix([
    [np.cos(y), -np.sin(y), 0],
    [np.sin(y), np.cos(y), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [np.cos(p), 0, np.sin(p)],
    [0, 1, 0],
    [-np.sin(p), 0, np.cos(p)]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, np.cos(r), -np.sin(r)],
    [0, np.sin(r), np.cos(r)]
    ])

    R = np.eye(4)
    R[:3, :3] = yawMatrix * pitchMatrix * rollMatrix

    return R

def make_matrix_xyz(x=0,y=0,z=0):
    
    T = np.eye(4)
    T[0,3] = x
    T[1,3] = y
    T[2,3] = z
    return T

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

#     assert(isRotationMatrix(R))

    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
