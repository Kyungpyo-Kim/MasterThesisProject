import os
import sys
import time , datetime

import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib

from tqdm import trange, tqdm_notebook

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util


# """ argument parser """
# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='model', help='Model name')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
# parser.add_argument('--pre_fix', default='', help='Pre-fix [default: pre-fix]')
# parser.add_argument('--data_dir', default='', help='all data directory')
# parser.add_argument('--data_train_dir', default='../data/train', help='Training data dir [default: ../data/train]')
# parser.add_argument('--data_test_dir', default='../data/test', help='Test data dir [default: ../data/test]')
# parser.add_argument('--num_point', type=int, default=512, help='Point Number [256/512/1024/2048] [default: 1024]')
# parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
# parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
# parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
# parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
# FLAGS = parser.parse_args()

# BATCH_SIZE = FLAGS.batch_size
# NUM_POINT = FLAGS.num_point
# MAX_EPOCH = FLAGS.max_epoch
# BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
# MOMENTUM = FLAGS.momentum
# OPTIMIZER = FLAGS.optimizer
# DECAY_STEP = FLAGS.decay_step
# DECAY_RATE = FLAGS.decay_rate


# """ model and training file """
# MODEL = importlib.import_module(FLAGS.model) # import network module
# MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
# LOG_DIR = FLAGS.log_dir
# if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    
# os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
# os.system(  'cp %s %s' % ( os.path.join(BASE_DIR, 'train.py'), LOG_DIR )  ) # bkp of train procedure


# """ log """
# LOG_FOUT = open(os.path.join(LOG_DIR, 'log_all_train.txt'), 'a')
# LOG_FOUT_data_path = open(os.path.join(FLAGS.data_dir, 'log_train.txt'), 'w')

# """ learning parameters """
# MAX_NUM_POINT = 512
# NUM_CLASSES = 8

# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP)
# BN_DECAY_CLIP = 0.99

# HOSTNAME = socket.gethostname()


# """ import train/test data set """
# TRAIN_FILES = []
# TEST_FILES = []

# if FLAGS.data_dir == '':
#     TRAIN_FILES = provider.getDataAllFiles( \
#         os.path.abspath(os.path.join(BASE_DIR, FLAGS.data_train_dir)))
#     TEST_FILES = provider.getDataAllFiles(\
#         os.path.abspath(os.path.join(BASE_DIR, FLAGS.data_test_dir)))
    
# else:
#     all_data_path = FLAGS.data_dir
    
#     all_data_path_list = [os.path.join(all_data_path, d) for d in os.listdir(all_data_path) if os.path.isdir(os.path.join(all_data_path, d))]
    
#     for p in all_data_path_list:
        
#         file_list = [os.path.join(p, f) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f)) and f.split('.')[-1] == 'h5']
        
#         file_list = sorted(file_list)
        
#         num_train = int (  float( len(file_list) ) * 0.7  )
#         num_test = len(file_list) - num_train
        
#         TRAIN_FILES.extend(file_list[:num_train])
#         TEST_FILES.extend(file_list[num_train:num_train + num_test])

#     """ backup train/test data """
#     backup_data_train_path = os.path.join(FLAGS.data_dir, 'train')
#     if not os.path.exists(backup_data_train_path): os.mkdir(backup_data_train_path)
#     else: os.system( "rm -r %s/*" % (backup_data_train_path) )
               
#     backup_data_test_path = os.path.join(FLAGS.data_dir, 'test')
#     if not os.path.exists(backup_data_test_path): os.mkdir(backup_data_test_path)
#     else: os.system( "rm -r %s/*" % (backup_data_test_path) )
            
#     for p in TRAIN_FILES:
#         os.system( "cp %s %s" % (p, backup_data_train_path) )
    
#     for p in TEST_FILES:
#         os.system( "cp %s %s" % (p, backup_data_test_path) )
    
        
# NUM_TO_LABEL = ['unknown', 'car', 'bus', 'bike', 'pedestrian', 'tree', 'building']
   


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        
        train_writer = None
        test_writer = None
        
        if FLAGS.data_dir == '' : train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        else: train_writer = tf.summary.FileWriter(backup_data_train_path, sess.graph)
            
        if FLAGS.data_dir == '' : test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        else: test_writer = tf.summary.FileWriter(backup_data_test_path)

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            print " Train one epoch %3d / %3d" % (epoch+1, MAX_EPOCH)
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer, epoch)
            print " Evaluate one epoch %3d / %3d" % (epoch+1, MAX_EPOCH)
            sys.stdout.flush()
            eval_one_epoch(sess, ops, test_writer, epoch)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                if FLAGS.data_dir == '' : save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                else: save_path = saver.save(sess, os.path.join(all_data_path, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
                
        if FLAGS.data_dir == '' : save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
        else: save_path = saver.save(sess, os.path.join(all_data_path, "model.ckpt"))
        log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer, epoch):
    """ ops: dict mapping from string to tf ops """                
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
#     for fn in range(len(TRAIN_FILES)):
#     for fn in tqdm_notebook(len(TRAIN_FILES)):
    for fn in trange(len(TRAIN_FILES)):
#         log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
#         log_string(  'batch_mean_loss: %f  batch_accuracy: %f' % ( loss_sum / float(num_batches) , total_correct / float(total_seen) )  )

        
def eval_one_epoch(sess, ops, test_writer, epoch):
    """ ops: dict mapping from string to tf ops """       
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_detect_class = [0 for _ in range(NUM_CLASSES)]
    total_detect_true_class = [0 for _ in range(NUM_CLASSES)]
    
#     for fn in range(len(TEST_FILES)):
#     for fn in tqdm_notebook(len(TEST_FILES)):
    for fn in trange(len(TEST_FILES)):
#         log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            for i in range(len(pred_val)):
                l = pred_val[i]
                total_detect_class[l] += 1
                total_detect_true_class[l] += (l == current_label[start_idx + i])
                
            
    log_string(  '[Epoch %d] eval mean loss: %f' % (  epoch, loss_sum / float(total_seen)  )   )
    log_string(  '[Epoch %d] eval accuracy: %f'% (  epoch, total_correct / float(total_seen)  )   )
    log_string(  '[Epoch %d] eval avg class acc: %f' % (  epoch, np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))  )   )
    for i_cls in range(NUM_CLASSES):
        if not total_seen_class[i_cls] == 0:
            log_string(  '[Epoch %d] eval indivisual [%s] class recall: %f' % (  epoch, NUM_TO_LABEL[i_cls], float(total_correct_class[i_cls])/float(total_seen_class[i_cls])))    
        if not total_detect_class[i_cls] == 0:
            log_string(  '[Epoch %d] eval indivisual [%s] class precision: %f' % (  epoch, NUM_TO_LABEL[i_cls], float(total_detect_true_class[i_cls])/float(total_detect_class[i_cls])))    
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
    LOG_FOUT_data_path.close()
