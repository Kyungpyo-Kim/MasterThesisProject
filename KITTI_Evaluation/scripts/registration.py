import numpy as np
from sklearn.neighbors import NearestNeighbors

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
        

    return T, R, t
   

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()
    
def icpMatchingPointNum(A, B, 
        init_pose = None, 
        max_iterations = 20, 
        tolerance = 0.001, 
        rm_outlier_dist = 1e3):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
        converged: Convergence of icp algorithm (mean error within the tolerance)
    '''


    matching_pt_num = 0.0
    
    if A.shape[0] == 0 or B.shape[0] == 0:
      T = np.eye(A.shape[1] + 1)
      return T, None, None, None, 0.0, 0.0
        
    A, B = random_upsampling(A, B)


    assert A.shape == B.shape
    assert max_iterations > 0
    
    
    converged = False
    mean_error = tolerance

    # get number of dimensions
    m = A.shape[1]
        
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    
    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
        
    for i in range(max_iterations):

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        
        # outlier remove
        idx_src = np.arange(B.shape[0])
        idx_dst = indices

        cond = distances < rm_outlier_dist
        
        if not np.sum(cond) > 0:
          break
        
        idx_src = idx_src[cond]
        idx_dst = indices[cond]
        dist = distances[cond]
        
        # remove duplicated elements
        idx_dst, i_src = np.unique(idx_dst, return_index = True)
        idx_src = idx_src[i_src]
        dist = dist[i_src]

        """
        for simulation
        """
        T,_,_ = best_fit_transform(src[:m,idx_src].T, dst[:m,idx_dst].T)
        
        # update the current source
        src = np.dot(T, src)
            
        # check error
        mean_error = np.max(np.abs(distances[cond]))
        if mean_error < tolerance:
            converged = True
            break
        
        matching_pt_num = np.sum(cond)

    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i, converged, mean_error, matching_pt_num    
  

def random_upsampling(A, B):

  assert A.shape[1] == B.shape[1]
  
  n1 = A.shape[0]
  n2 = B.shape[0]
  
  n = n1>n2 and n1 or n2
    
  if not n1 == n: # A
    sample = np.random.choice(n1, n-n1)
    a = A[sample]    
    AA = np.concatenate([A, a], axis=0)
    
  else:
    AA = A
    
  if not n2 == n: # B
    sample = np.random.choice(n2, n-n2)
    b = B[sample]    
    BB = np.concatenate([B, b], axis=0)
    
  else:
    BB = B
    
  return AA, BB


def layered_registration(source, target, iteration, prev_source = None):
        
    """ parameters """
    # layer boundary condition
    l1_bnd = 0.6
    l2_bnd = l1_bnd * 2.0
    l3_bnd = l1_bnd * 3.0
    h_bnd = 0.2

    tolerance = 0.1

    """ init pose estimation """ 
    # prev point cloud
    init_pose = np.eye(3)

    if not prev_source == None:
        ps = prev_source.copy()

    """ layer-based sampling """
    # layer1
    tl1 = target.copy()
    tl1 = tl1[ tl1[:,2] < l1_bnd]

    l1_s_bnd = np.mean( tl1[:,2] )

    sl1 = source.copy()
  
    sl1 = sl1[ sl1[:,2] < l1_s_bnd + h_bnd ]
    sl1 = sl1[ sl1[:,2] > l1_s_bnd - h_bnd ]

    # layer2
    tl2 = target.copy()
    tl2 = tl2[ tl2[:,2] < l2_bnd ]
    tl2 = tl2[ tl2[:,2] > l1_bnd ] 

    l2_s_bnd = np.mean( tl2[:,2] )

    sl2 = source.copy()
    sl2 = sl2[ sl2[:,2] < l2_s_bnd + h_bnd ]
    sl2 = sl2[ sl2[:,2] > l2_s_bnd - h_bnd ]
    
    # layer3
    tl3 = target.copy()
    tl3 = tl3[ tl3[:,2] < l3_bnd ]
    tl3 = tl3[ tl3[:,2] > l2_bnd ] 

    l3_s_bnd = np.mean( tl3[:,2] )

    sl3 = source.copy()
    sl3 = sl3[ sl3[:,2] < l3_s_bnd + h_bnd ]
    sl3 = sl3[ sl3[:,2] > l3_s_bnd - h_bnd ]

    # layer4
    tl4 = target.copy()
    tl4 = tl4[ tl4[:,2] > l3_bnd ] 

    l4_s_bnd = np.mean( tl4[:,2] )

    sl4 = source.copy()
    sl4 = sl4[ sl4[:,2] < l4_s_bnd + h_bnd ]
    sl4 = sl4[ sl4[:,2] > l4_s_bnd - h_bnd ]
            
    # matching1
    mat_tf1, _, _, _, me1, mat_pt_num1 = icpMatchingPointNum(sl1[:, :2],
                                    tl1[:, :2],
                                    init_pose = init_pose,
                                    max_iterations = 100,
                                    tolerance = tolerance,
                                    rm_outlier_dist = 0.2)

    # matching2
    mat_tf2, _, _, _, me2, mat_pt_num2 = icpMatchingPointNum(sl2[:, :2],
                                    tl2[:, :2],
                                    init_pose = init_pose,
                                    max_iterations = 100,
                                    tolerance = tolerance,
                                    rm_outlier_dist = 0.2)
    # matching3
    mat_tf3, _, _, _, me3, mat_pt_num3 = icpMatchingPointNum(sl3[:, :2],
                                    tl3[:, :2],
                                    init_pose = init_pose,
                                    max_iterations = 100,
                                    tolerance = tolerance,
                                    rm_outlier_dist = 0.2)
    # matching4
    mat_tf4, _, _, _, me4, mat_pt_num4 = icpMatchingPointNum(sl4[:, :2],
                                    tl4[:, :2],
                                    init_pose = init_pose,
                                    max_iterations = 100,
                                    tolerance = tolerance,
                                    rm_outlier_dist = 0.2)


    alpha = 0.9
    c = 1.
    n_layer = np.array( [ tl1.shape[0], tl2.shape[0], tl3.shape[0], tl4.shape[0] ] )
    e_layer = np.array( [ 1./(c + me1), 1./(c + me2), 1./(c + me3), 1./(c + me4)] )

    w_layer = alpha * n_layer / np.sum(n_layer) + (1. - alpha) * e_layer / np.sum(e_layer)

    mat_tf =  mat_tf1 * w_layer[0] \
            + mat_tf2 * w_layer[1] \
            + mat_tf3 * w_layer[2] \
            + mat_tf4 * w_layer[3]

    return None, mat_tf, None, np.mean([me1, me2, me3, me4])