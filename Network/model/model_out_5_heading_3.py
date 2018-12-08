# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util

from transform_nets import input_transform_net, feature_transform_net


def placeholder_inputs(batch_size, num_point):
    
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    class_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    heading_labels_pl = tf.placeholder(tf.float32, shape=(batch_size))
    
    return pointclouds_pl, class_labels_pl, heading_labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 4, activation_fn=None, scope='fc3')

    return net, end_points


"""
Total loss
"""
def get_loss(pred, class_label, heading_label, end_points, reg_weight=0.001, h_reg_weight=0.1):
    """ 
    pred: B * (NUM_CLASSES + heading), 
    label: B
    """
   
    classify_loss = get_classify_loss(pred, class_label)
    heading_loss = get_heading_loss(pred, heading_label)
    mat_diff_loss = get_mat_diff_loss(end_points)

    tf.summary.scalar('classify_loss', classify_loss)
    tf.summary.scalar('heading_loss', heading_loss)
    tf.summary.scalar('mat_loss', mat_diff_loss)
    
    return classify_loss + mat_diff_loss * reg_weight + heading_loss * h_reg_weight


"""
Classification loss
"""
def get_classify_loss(pred, class_label):
    
    batch_size = pred.get_shape()[0].value
    class_pred = tf.slice(pred, [0,0], [batch_size,3])   
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_pred, labels=class_label)
    
    classify_loss = tf.reduce_mean(loss)
    
    return classify_loss
    

"""
Heading loss
"""
def get_heading_loss(pred, heading_label):    
    # reference: http://pythonkim.tistory.com/17 [파이쿵]
    #regression_loss = -tf.reduce_mean(heading_label * tf.log(heading_pred) + (1 - heading_label) * tf.log(1 - heading_pred))
    batch_size = pred.get_shape()[0].value
    heading_pred = tf.slice(pred, [0,3], [batch_size,1])
    
    regression_loss = tf.reduce_mean(tf.square(heading_pred - heading_label))
    
    return regression_loss
    
    
"""
Matrix regression loss
"""
def get_mat_diff_loss(end_points):
    
    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    
    return mat_diff_loss
    


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        class_label = tf.zeros((32,),  tf.int32)
        heading_label = tf.zeros((32,))
        pred, end_points = get_model(inputs, tf.constant(True))
        get_loss(pred, class_label, heading_label, end_points)
    print(pred.get_shape())
