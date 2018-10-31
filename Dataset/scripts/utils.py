# -*- coding: utf-8 -*-

# Add modules
import os
import pickle
import Tkinter as tk # python 2
import tkFileDialog as filedialog
import struct
import numpy as np
import pcl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import cv2
import h5py

## Add constants
Geod_a = 6378137.0
Geod_e2 = 0.00669437999014
RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180
mm2m = 1 / 1000
XPixelMeter = 0.024699619563463528


# Select path
def FnSelectPath(title):

    # Remove tk windows
    root = tk.Tk()
    root.withdraw()

    # Load pickle
    pickle_logging_path = os.path.dirname(os.path.abspath(__file__)) + '/prev_logging_path.p'
    try:
        with open(pickle_logging_path,'rb') as f:
            pickle_for_load=pickle.load(f)
    
        # Get file path 
        options = {}
        options['initialdir'] = pickle_for_load['prev_logging_path']
        options['title'] = title
        logging_path = tk.filedialog.askdirectory(**options)

    except:
        print('Fail to read the previous logging path')

        # Get file path 
        options = {}
        options['title'] = title
        logging_path = tk.filedialog.askdirectory(**options)

    
    # save pickle
    pickle_for_save = {'prev_logging_path':logging_path}
    with open(pickle_logging_path,'wb') as f:
       pickle.dump(pickle_for_save,f)
    
    # return
    return logging_path


def FnGetFileList(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return onlyfiles


def FnParsePointCloudFile(file):
    data_list = []    
    # Open file    
    with open(file, "rb") as f:
        while True:
            # read 
            tmp_timestamp=f.read(8)
            if tmp_timestamp==b'': break
                
            # Define row
            row=[]
            # Get timestamp
            timestamp = struct.unpack("<Q",tmp_timestamp)[0]
            row.append(timestamp)
            
            # Get number of points
            num_point = struct.unpack("<i",f.read(4))[0]
            row.append(num_point)
            
            ## Get current pointer
            row.append(f.tell())
        
            # Add row
            data_list.append(row)
            
            ## Jump to next point
            f.seek(num_point*8*6,1)
               
    return data_list


def FnParsePointCloud(file, num_point, file_pointer):
    with open(file, "rb") as f:
        f.seek(file_pointer)
        strFormat = "<"+str(num_point * 6) + "d"
        array1d = np.array(struct.unpack(strFormat, f.read(num_point * 8 * 6)), dtype=np.float32)
        return np.reshape(array1d, (num_point,-1))


def FnLLH2ENU(RefWGS84, TargetWGS84):
    # Latitude, longitude, height
    dKappaLat = 0
    dKappaLon = 0
    
    if TargetWGS84.shape[1] == 2:
        height = np.zeros(TargetWGS84.shape[0])
    elif TargetWGS84.shape[1] == 3:
        height = TargetWGS84[:,2]

    dKappaLat = FnKappaLat(RefWGS84[0], height)
    dKappaLon = FnKappaLon(RefWGS84[0], height)
        
    dEast_m = np.divide((TargetWGS84[:,1] - RefWGS84[1]), dKappaLon)
    dNorth_m = np.divide((TargetWGS84[:,0] - RefWGS84[0]), dKappaLat)
    dHeight_m = height

    TargetENU = dEast_m
    TargetENU = np.vstack((TargetENU, dNorth_m))

    TargetENU = np.vstack((TargetENU, dHeight_m))
    TargetENU = np.transpose(TargetENU)

    return TargetENU


def FnENU2LLH(RefWGS84, TargetENU):
    # Latitude, longitude, height
    dKappaLat = 0
    dKappaLon = 0
    
    if TargetENU.shape[1] == 2:
        height = np.zeros(TargetENU.shape[0])
    elif TargetENU.shape[1] == 3:
        height = TargetENU[:,2]

    dKappaLat = FnKappaLat(RefWGS84[0], height)
    dKappaLon = FnKappaLon(RefWGS84[0], height)

    dLatitude_deg = RefWGS84[0] + np.multiply(dKappaLat, TargetENU[:,1])
    dLongitude_deg = RefWGS84[1] + np.multiply(dKappaLon, TargetENU[:,0])
    dHeight_m = height
    
    TargetLLH = dLatitude_deg
    TargetLLH = np.vstack((TargetLLH, dLongitude_deg))
    TargetLLH = np.vstack((TargetLLH, dHeight_m))
    TargetLLH = np.transpose(TargetLLH)

    return TargetLLH


def FnKappaLat(dLatitude, dHeight):
    dKappaLat = 0
    Denominator = 0
    dM = 0

    Denominator = np.sqrt(1 - Geod_e2 * pow(np.sin(dLatitude * DEG2RAD), 2))
    dM = Geod_a * (1 - Geod_e2) / pow(Denominator, 3)
    
    dKappaLat = 1 / (dM + dHeight) * RAD2DEG
    
    return dKappaLat


def FnKappaLon(dLatitude, dHeight):
    dKappaLon = 0
    Denominator = 0
    dN = 0
    
    Denominator = np.sqrt(1 - Geod_e2 * pow(np.sin(dLatitude * DEG2RAD), 2))
    dN = Geod_a / Denominator
    
    dKappaLon = 1 / ((dN + dHeight) * np.cos(dLatitude * DEG2RAD)) * RAD2DEG
    
    return dKappaLon
    
def GetPointCloudFromFile(file_path, file_list, idx):

    pt_np = FnParsePointCloud(file_path,
                                        file_list[idx][1],
                                        file_list[idx][2])

    pt_np_xyzrgb = np.zeros((pt_np.shape[0], 3), dtype=np.float32)

    pt_np_xyzrgb[:, :3] = pt_np[:, :3]
    pt_np_xyzrgb = pt_np_xyzrgb[pt_np_xyzrgb[:,2] > 0.5, :3]

    cloud = pcl.PointCloud()
    cloud.from_array(pt_np_xyzrgb)

    return cloud

def display_point_cloud(pts):
    fig = plt.figure()
    
    # display point cloud
    ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    
    # display center point
    cp = np.mean(pts, axis = 0)
    ax.scatter(cp[:,0], cp[:,1], cp[:,2], 'r')
        
    plt.show()

def resample_point_cloud(pts, resample_ratio):
    step = int(1./resample_ratio)
    idx = range(0, len(pts), step)
    return pts[idx]


def display_point_cloud_8_image(pts, ratio, image):
    
    fig = plt.figure()
    num_plot = 331
    
    sub_plot_title = ['Front-Left', 'Front-Center', 'Front-Right', 'Mid-Left', 'Mid-Right', 'Rear-Left', 'Rear-Center', 'Rear-Right']
    
    for i in range(8):
        
        j = i
        if j >=4 : j +=1 
        
        resample_data = resample_point_cloud(pts[i],ratio)
        
        ax = fig.add_subplot(331 + j, projection='3d')    
        ax = display_point_cloud_box_ax(ax, resample_data)
#         ax.scatter(resample_data[:,0], resample_data[:,1], resample_data[:,2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(sub_plot_title[i], fontweight='bold')
#         ax.set_title(su)
    
        # make axis equal
        p_min = np.min(pts[i], axis=0)
        p_max = np.max(pts[i], axis=0)
        max_range = np.array([p_max[0]-p_min[0], p_max[1]-p_min[1], p_max[2]-p_min[2]]).max() / 2.0

        mid_x = (p_max[0]+p_min[0]) * 0.5
        mid_y = (p_max[1]+p_min[1]) * 0.5
        mid_z = (p_max[2]+p_min[2]) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.invert_xaxis()
        ax.invert_yaxis()
        
    ax = fig.add_subplot(335)    
    ax.imshow(image)
    ax.axis("off")
    
    plt.show()
    
def display_point_cloud_box(pts):
    fig = plt.figure()
    
    # display point cloud
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    
    # display center point
    cp = np.mean(pts, axis = 0)
    ax.scatter(cp[0], cp[1], cp[2], c='r', s=50)
    
    # display bounding box   
    cube_definition_array = [
        np.array(list(item))
        for item in find_cube_definition(pts)
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.01))
    
    ax.add_collection3d(faces)
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)
    
    # make axis equal
    p_min = np.min(points, axis=0)
    p_max = np.max(points, axis=0)
    max_range = np.array([p_max[0]-p_min[0], p_max[1]-p_min[1], p_max[2]-p_min[2]]).max() / 2.0

    mid_x = (p_max[0]+p_min[0]) * 0.5
    mid_y = (p_max[1]+p_min[1]) * 0.5
    mid_z = (p_max[2]+p_min[2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    
    plt.show()  
    
    
def display_point_cloud_box_ax(ax, pts):
    
    # display point cloud
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    
    # display center point
    cp = np.mean(pts, axis = 0)
    ax.scatter(cp[0], cp[1], cp[2], c='r', s=50)
    
    # display bounding box   
    cube_definition_array = [
        np.array(list(item))
        for item in find_cube_definition(pts)
    ]
    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.05))
    
    ax.add_collection3d(faces)
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    # make axis equal
    p_min = np.min(points, axis=0)
    p_max = np.max(points, axis=0)
    max_range = np.array([p_max[0]-p_min[0], p_max[1]-p_min[1], p_max[2]-p_min[2]]).max() / 2.0

    mid_x = (p_max[0]+p_min[0]) * 0.5
    mid_y = (p_max[1]+p_min[1]) * 0.5
    mid_z = (p_max[2]+p_min[2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax


def find_cube_definition(pts):
    p_min = np.min(pts, axis = 0)
    p_max = np.max(pts, axis = 0)
    return [ (p_min[0],p_min[1],p_min[2]), 
            (p_max[0],p_min[1],p_min[2]), 
            (p_min[0],p_max[1],p_min[2]), 
            (p_min[0],p_min[1],p_max[2])]


# Data normalization and resampling
def NormalizeResample(data, num_sample):
    """ data is in N x ...
    we want to keep num_samplexC of them.
    if N > num_sample, we will randomly keep num_sample of them.
    if N < num_sample, we will randomly duplicate samples.
    """
  
    ## normalizing   
    x_min = float(data[:,0].min())
    x_max = float(data[:,0].max())
    y_min = float(data[:,1].min())
    y_max = float(data[:,1].max())
    z_min = float(data[:,2].min())
    z_max = float(data[:,2].max())
    
    scale_val = np.max([ x_max - x_min , y_max - y_min , z_max - z_min ])
    offset_val = np.min([x_min, y_min, z_min])
      
    data[:,0] = data[:,0] - offset_val
    data[:,1] = data[:,1] - offset_val
    data[:,2] = data[:,2] - offset_val
      
    data[:,0] = data[:,0] / float(scale_val)
    data[:,1] = data[:,1] / float(scale_val)
    data[:,2] = data[:,2] / float(scale_val)
                 
    ## resampling
    N = data.shape[0]
    if (N == num_sample):
        return data
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...]
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0)
    
""" save funcation """

def save_h5_file(data_path, data, label):
                 
    if os.path.exists(data_path):
        os.system('rm {}'.format(data_path))

    h5_train = h5py.File(data_path)
    
    h5_train.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_train.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_train.close()
    print "[Generate] path: {}, \ndata shape: {}, label shape: {}".format(data_path, str(data.shape), str(label.shape) )

data_dtype = 'float32'
label_dtype = 'uint8'

def save_h5_files(data, label, out_path, data_dtype, label_dtype):

    print "Total data:", len(data)
    print "Total label:", len(label)
    
    label_list = [0, 1, 2] ## unknown, cars, trucks
    
    np_data = np.array(data)
    np_label = np.array(label)
    
    ## shuffle data
    idx = np.arange(len(data))
    
    np.random.shuffle(idx)

    np_data = np_data[idx]
    np_label = np_label[idx]
    
    ## dividing data
    ## train:6 test:2 vali:2
    
    data_train_list =[]
    data_test_list =[]
    data_vali_list =[]
    
    label_train_list =[]
    label_test_list =[]
    label_vali_list =[]
    
    for cls in label_list:
        
        print cls
        total_num =  np.sum(np_label == cls)
        train_num = int (total_num * 0.6)
        test_num = int (total_num * 0.2)
        vali_num = total_num - train_num - test_num
        
        np_data_filtered = np_data[np_label == cls]
        np_label_filtered = np_label[np_label == cls]
        
        data_train = np_data_filtered[:train_num]
        data_test = np_data_filtered[train_num: train_num + test_num]
        data_vali = np_data_filtered[train_num + test_num:]
        
        label_train = np_label_filtered[:train_num]
        label_test = np_label_filtered[train_num: train_num + test_num]
        label_vali = np_label_filtered[train_num + test_num:]
        
        print "np_data_filtered.shape", np_data_filtered.shape
        print "data_train.shape", data_train.shape
        print "data_test.shape", data_test.shape
        print "data_vali.shape", data_vali.shape
        
        data_train_list.append(data_train)
        data_test_list.append(data_test)
        data_vali_list.append(data_vali)
        label_train_list.append(label_train)
        label_test_list.append(label_test)
        label_vali_list.append(label_vali)
    
    
    data_train = data_train_list[0]
    for i in range( len(data_train_list) - 1 ):
        data_train = np.concatenate( (data_train, data_train_list[i+1]) , axis = 0 )
    
    data_test = data_test_list[0]
    for i in range( len(data_test_list) - 1 ):
        data_test = np.concatenate( (data_test, data_test_list[i+1]) , axis = 0 )
    
    data_vali = data_vali_list[0]
    for i in range( len(data_vali_list) - 1 ):
        data_vali = np.concatenate( (data_vali, data_vali_list[i+1]) , axis = 0 )
    
    label_train = label_train_list[0]
    for i in range( len(label_train_list) - 1 ):
        label_train = np.concatenate( (label_train, label_train_list[i+1]) , axis = 0 )
    
    label_test = label_test_list[0]
    for i in range( len(label_test_list) - 1 ):
        label_test = np.concatenate( (label_test, label_test_list[i+1]) , axis = 0 )
    
    label_vali = label_vali_list[0]
    for i in range( len(label_vali_list) - 1 ):
        label_vali = np.concatenate( (label_vali, label_vali_list[i+1]) , axis = 0 )
    
    
    train_path = os.path.join(out_path, 'train.h5')
    save_h5_file(train_path, data_train, label_train)
    
    test_path = os.path.join(out_path, 'test.h5')
    save_h5_file(test_path, data_test, label_test)
    
    vali_path = os.path.join(out_path, 'vali.h5')
    save_h5_file(vali_path, data_vali, label_vali)
    
    
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)