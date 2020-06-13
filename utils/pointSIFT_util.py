"""
wrappers for pointSIFT module
Author: Jiang Mingyang
Email: jmydurant@sjtu.edu.cn
"""
from utils.tf_ops.pointSIFT_op.pointSIFT_op import pointSIFT_select, pointSIFT_select_two, pointSIFT_select_four
from utils.tf_ops.grouping.tf_grouping import group_point, query_ball_point, knn_point
from utils.tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from utils.tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
#import utils.tf_util as tf_util
import helper_tf_util
import tensorflow as tf
import numpy as np


def gather_neighbour(pc, neighbor_idx):
    # gather the coordinates or features of neighboring points
    batch_size = tf.shape(pc)[0] #tensor
    num_points = tf.shape(pc)[1]
    d = pc.get_shape()[2].value # int 3
    index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
    features = tf.batch_gather(pc, index_input)
    features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d]) #shape(?,?,?,3) 
    return features
    
def pointSIFT_group(radius, xyz, points, use_xyz=True):
    #import pdb
    #pdb.set_trace()
    idx = pointSIFT_select(xyz, radius) # 输入(batch_size,npoint,3) 输出(batch_size,npoint,8) 输出8个象限的点的idx 
    #grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3) 输出8个象限的点坐标 
    grouped_xyz = gather_neighbour(xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])  # translation normalization 将参考点复制8次，相减得到相对坐标
    if points is not None:
        #grouped_points = group_point(points, idx)  # (batch_size, npoint, 8, channel) 输出8个象限的点特征 points指代feature
        grouped_points = gather_neighbour(tf.squeeze(points, axis=2), idx)
        if use_xyz: # 是否将坐标加到特征后面
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    #输出：原始点坐标， 8个象限的点特征+坐标，8个象限点的idx，8个象限的点相对坐标
    return xyz, new_points, idx, grouped_xyz

def pointSIFT_group_with_idx(xyz, idx, points, use_xyz=True):
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 8, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return xyz, new_points, idx, grouped_xyz


def pointSIFT_group_two(radius, xyz, points, use_xyz=True):
    #import pdb
    #pdb.set_trace()
    idx = pointSIFT_select_two(xyz, radius) # 输入(batch_size,npoint,3) 输出(batch_size,npoint,16) 输出8个象限的点的idx 每个象限2个点
    #grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3) 输出8个象限的点坐标 
    grouped_xyz = gather_neighbour(xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 16, 1])  # translation normalization 将参考点复制16次，相减得到相对坐标
    if points is not None:
        #grouped_points = group_point(points, idx)  # (batch_size, npoint, 16, channel) 输出8个象限的点特征 points指代feature
        grouped_points = gather_neighbour(tf.squeeze(points, axis=2), idx)
        if use_xyz: # 是否将坐标加到特征后面
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 16, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    #输出：原始点坐标， 8个象限的点特征+坐标，8个象限点的idx，8个象限的点相对坐标
    return xyz, new_points, idx, grouped_xyz

def pointSIFT_group_two_with_idx(xyz, idx, points, use_xyz=True):
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 16, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 16, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 16, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 16, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return xyz, new_points, idx, grouped_xyz


def pointSIFT_group_four(radius, xyz, points, use_xyz=True):
    idx = pointSIFT_select_four(xyz, radius)
    #grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 32, 3)
    grouped_xyz = gather_neighbour(xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 32, 1])  # translation normalization
    if points is not None:
        #grouped_points = group_point(points, idx)  # (batch_size, npoint, 32, channel)
        grouped_points = gather_neighbour(tf.squeeze(points, axis=2), idx)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return xyz, new_points, idx, grouped_xyz

def pointSIFT_group_four_with_idx(xyz, idx, points, use_xyz=True):
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 32, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 32, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, 32, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 32, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return xyz, new_points, idx, grouped_xyz

def pointSIFT_module(xyz, points, radius, out_channel, is_training, bn_decay=None, scope='point_sift', bn=True, use_xyz=True, use_nchw=False):
    #data_format = 'NCHW' if use_nchw else 'NHWC' # 默认NHWC channel在后面,randlanet tf_utils没有这个参数，
    with tf.variable_scope(scope) as sc:
        # Grouping
        new_xyz, new_points, idx, grouped_xyz = pointSIFT_group(radius, xyz, points, use_xyz) #输出：原始点坐标， 8个象限的点特征+相对坐标，8个象限点的idx，8个象限的点相对坐标

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
        for i in range(3):
            new_points = helper_tf_util.conv2d(new_points, out_channel, [1, 2],# kernel size
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay)
        # add fc
        new_points = helper_tf_util.conv2d(new_points, out_channel, [1, 1],
                                    padding='VALID', stride=[1, 1],
                                    bn=bn, is_training=is_training,
                                    scope='conv_fc', bn_decay=bn_decay)
        if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointSIFT_res_module(xyz, points, radius, out_channel, is_training, bn_decay, scope='point_sift', bn=True, use_xyz=True, same_dim=False, merge='add'):
    #data_format = 'NHWC'
    with tf.variable_scope(scope) as sc:
        # conv1
        _, new_points, idx, _ = pointSIFT_group(radius, xyz, points, use_xyz=use_xyz)

        for i in range(3):
            new_points = helper_tf_util.conv2d(new_points, out_channel, [1, 2],
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='c0_conv%d' % (i), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2])
        # conv2
        _, new_points, idx, _ = pointSIFT_group_with_idx(xyz, idx=idx, points=new_points, use_xyz=use_xyz)

        for i in range(3):
            if i == 2:
                act = None
            else:
                act = tf.nn.relu
            new_points = helper_tf_util.conv2d(new_points, out_channel, [1, 2],
                                        padding='VALID', stride=[1, 2],
                                        bn=bn, is_training=is_training,
                                        scope='c1_conv%d' % (i), bn_decay=bn_decay,
                                        activation_fn=act)
        new_points = tf.squeeze(new_points, [2])
        points = tf.squeeze(points, [2])
        # residual part..
        if points is not None:
            if same_dim is True:
                points = helper_tf_util.conv1d(points, out_channel, 1, padding='VALID', bn=bn, is_training=is_training, scope='merge_channel_fc', bn_decay=bn_decay)
            if merge == 'add':
                new_points = new_points + points
            elif merge == 'concat':
                new_points = tf.concat([new_points, points], axis=-1)
            else:
                print("ways not found!!!")
        new_points = tf.nn.relu(new_points)
        return xyz, new_points, idx


"""
PointNet++ layers
Author: Charles R. Qi
Date: November 2017
"""


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope,
                       bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    #data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
        for i, num_out_channel in enumerate(mlp):
            new_points = helper_tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay)
        if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        # Pooling in Local Regions
        if pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling == 'weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(exp_dists, axis=2,
                                                    keep_dims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0, 3, 1, 2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = helper_tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1, 1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d' % (i), bn_decay=bn_decay)
            if use_nchw: new_points = tf.transpose(new_points, [0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope,
                           bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    #data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0, 3, 1, 2])
            for j, num_out_channel in enumerate(mlp_list[i]):
                grouped_points = helper_tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                                padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d' % (i, j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0, 2, 3, 1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = helper_tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d' % (i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1
    
