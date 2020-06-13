from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
from utils.pointSIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_fp_module, pointnet_sa_module


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder 结果存放路径
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers # 从inputs中取出各元素
            self.inputs['xyz'] = flat_inputs[:num_layers] # 原始batch里的点云坐标 （五层）iter shape=(?,?,3)
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers] # 原始点云k近邻
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers] # k近邻对应原始点云下采样
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers] # 原始点云下采样后上采样
            self.inputs['features'] = flat_inputs[4 * num_layers] # 特征
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1] # 标签
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2] # 点的index
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3] # 点云的index

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name) #得到每个类别的点的数量 对于S3DIS为13维
            #[[ 4.97221634,  5.76365047,  3.51712115, 26.95745666, 24.88543334,
            #21.88936609, 13.81691932, 18.19796956, 15.99252438, 40.37542513,
            #11.23598421, 30.91618524,  7.03602697]]
            self.Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a') # log_train_S3DISArea_1.txt

        with tf.variable_scope('layers'): # 构建层
            self.logits, self.scene_logits = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.scene_logits = tf.reshape(self.scene_logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label)) # 将ignored点置为true

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool))) # 有效的点
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_scene_logits = tf.gather(self.scene_logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape 没太懂
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights) # 计算loss

        with tf.variable_scope('optimizer'): # 优化器
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'): # tensorboard
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto() # 1.记录设备指派情况； 2.自动选择运行设备 3.限制GPU资源使用
        c_proto.gpu_options.allow_growth = True # 动态申请显存
        self.sess = tf.Session(config=c_proto) # 动态申请显存
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph) # tensorboard写入文件
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out # d_out = [16, 64, 128, 256, 512]  # feature dimension 都乘以2
        feature = inputs['features'] # iter shape=(?,?,6) xyz rgb
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0') # 全连接层 fc0 输出维度8 shape(?,?,8) (batch_size,num_points,3)
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training)) # leaky_relu激活 shape(?,?,8)
        feature = tf.expand_dims(feature, axis=2) # 增加一个维度 shape(?,?,1,8) 用于K的加入

        # ###########################Encoder############################
        f_encoder_list = []
        #import pdb
        #pdb.set_trace()
        for i in range(self.config.num_layers): # 每一层都会构建,共5层
            radius = 0.1*(i+1) # 设置每层pointSIFT搜索范围
            merge_ = 'add'
            if i <= 1: # 前两个block通道数太小，不适合SEBlock manifold of interest
            #    merge_ = 'concat'
                f_encoder_i = self.dilated_res_PointSIFT_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training, radius=radius, merge=merge_) # 首先通过dilated_res_block (?,?,1,32) 128 156 512 1024
            else:
                f_encoder_i = self.dilated_res_PointSIFT_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training, radius=radius, merge=merge_) 
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i]) # 对特征随机采样 (?, ?/4, 1, 32) sub_sampling_ratio = [4, 4, 4, 4, 2] 整数除法
            feature = f_sampled_i # 会保存最后一层特征
            if i == 0:
                f_encoder_list.append(f_encoder_i) # (?,?,1,32) N 保存未下采样的
            f_encoder_list.append(f_sampled_i) # 之后的层下采样 总共有6个
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)  # 相当于MLP
        feature_cpy = feature
        # ###########################Scene Encoder############################
        
        for j in range(self.config.num_layers):
            f_scene_encoder = self.nearest_interpolation(feature_cpy, inputs['interp_idx'][-j - 1]) # 先上采样
            feature_cpy = f_scene_encoder            
        import pdb
        pdb.set_trace()
        f_scene_fc1 = helper_tf_util.conv2d(f_scene_encoder, 64, [1, 1], 'scene_fc1', [1, 1], 'VALID', True, is_training)
        f_scene_fc2 = helper_tf_util.conv2d(f_scene_fc1, 32, [1, 1], 'scene_fc2', [1, 1], 'VALID', True, is_training)
        f_scene_drop = helper_tf_util.dropout(f_scene_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_scene_fc3 = helper_tf_util.conv2d(f_scene_drop, self.config.num_classes, [1, 1], 'scene_fc3', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None) # (?,?,?,13)
        f_scene_out = tf.squeeze(f_scene_fc3, [2]) # (?,?,13)
        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1]) # 先上采样
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training) # 有跳连操作，也是充当MLP
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc3', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None) # (?,?,?,13)
        f_out = tf.squeeze(f_layer_fc3, [2]) # (?,?,13)
        f_out = tf.multiply(f_out, f_scene_out) 
        return f_out, f_scene_out

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.scene_logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate 更新学习率
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1)) # 混淆矩阵，计算多类别的precision和recall
                gt_classes += np.sum(conf_matrix, axis=1) # recall = tp / gt_classes
                positive_classes += np.sum(conf_matrix, axis=0) # precision = tp / positive_classes
                true_positive_classes += np.diagonal(conf_matrix) # tp

            except tf.errors.OutOfRangeError:
                break

        iou_list = [] # 混淆矩阵的IoU
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency 出现频率越小，loss weights越大
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    '''def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)# kernel[1,1,8,8] 输出shape(?,?,1,8)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training) # (?,?,1,16)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None) # (?,?,1,32)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training) # (?,?,1,32)
        return tf.nn.leaky_relu(f_pc + shortcut) # tf.nn.leaky_relu(f_pc + shortcut)'''
    
    
    def dilated_res_PointSIFT_block(self, feature, xyz, neigh_idx, d_out, name, is_training, radius, merge):
        '''
        embbed PointSIFT module into dilated_res_block
        '''
        #import pdb
        #pdb.set_trace()
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)# kernel[1,1,8,8] 输出shape(?,?,1,8)
        new_xyz, f_pc, idx = pointSIFT_res_module(xyz, f_pc, radius, d_out // 2, is_training, None, name + 'pointSIFT1', True, True, False, merge) # pointSIFT会自动把相对坐标和特征concatenate起来
        f_pc = helper_tf_util.conv2d(tf.expand_dims(f_pc, axis=2), d_out, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        
        new_xyz, f_pc, idx = pointSIFT_res_module(xyz, f_pc, radius, d_out, is_training, None, name + 'pointSIFT2', True, True, False, merge)
        f_pc = helper_tf_util.conv2d(tf.expand_dims(f_pc, axis=2), d_out * 2, [1, 1], name + 'mlp3', [1, 1], 'VALID', True, is_training)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training) # (?,?,1,32)
        return tf.nn.leaky_relu(f_pc + shortcut) # tf.nn.leaky_relu(f_pc + shortcut)
    
    
    
    def dilated_res_PointSIFT_SE_block(self, feature, xyz, neigh_idx, d_out, name, is_training, radius, merge):
        '''
        embbed PointSIFT module and SEBlock into dilated_res_block
        '''
        #import pdb
        #pdb.set_trace()
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)# kernel[1,1,8,8] 输出shape(?,?,1,8)
        new_xyz, f_pc, idx = pointSIFT_res_module(xyz, f_pc, radius, d_out // 2, is_training, None, name + 'pointSIFT1', True, True, False, merge) # pointSIFT会自动把相对坐标和特征concatenate起来
        f_pc = self.se_block(f_pc, d_out, name + 'att1', is_training) # (?,?,1,16)       
        
        new_xyz, f_pc, idx = pointSIFT_res_module(xyz, f_pc, radius, d_out, is_training, None, name + 'pointSIFT2', True, True, False, merge)
        f_pc = self.se_block(f_pc, d_out * 2, name + 'att2', is_training) #(?,?,1,32)        

        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training) # (?,?,1,32)
        return tf.nn.leaky_relu(f_pc + shortcut) # tf.nn.leaky_relu(f_pc + shortcut)
    
    
    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value #(?,?,1,8)取前面最后一个输出的维度  8
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx) # (?,?,?,10)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training) #(?,?,?,8)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx) # (?,?,?,8)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1) # (?,?,?,16)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training) # (?,?,1,8)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)#(?,?,?,8)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx) 
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training) # (?,?,1,16)
        return f_pc_agg
    
    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx) # [batch_size, num_points, tf.shape(neighbor_idx)[-1], d]) (?,?,?,3)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1]) # tensorflow中的tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。将 K*3复制N次
        #(?,?,1,3)->(?,?,k,3) 实际k unknown (?,?,?,3) 用于计算相对坐标
        relative_xyz = xyz_tile - neighbor_xyz 
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True)) # (?,?,?,1)
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1) # (?,?,?,10)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2) # (?, ?, 32)
        num_neigh = tf.shape(pool_idx)[-1] 
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1]) # (?,?) [B,N']
        pool_features = tf.batch_gather(feature, pool_idx) # (?, ?, 32)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d]) # (?, ?, ?,32)
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True) # (?, ?, 1,32)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0] #tensor
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value # int 3
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d]) #shape(?,?,?,3) 
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d]) # -1指第一维的shape由后面的计算出来 batch_size*num_points
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc') # (?,?,16)
        att_scores = tf.nn.softmax(att_activation, axis=1) # (?,?,16)
        f_agg = f_reshaped * att_scores # (?,?,16)
        f_agg = tf.reduce_sum(f_agg, axis=1) # (?,16)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d]) # (?,?,1,16)
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)#(?,?,1,16)
        return f_agg
    
    @staticmethod
    def se_block(feature_set, d_out, name, is_training):
        '''
        learn from SEBlock channel-wise attention
        '''
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        d = feature_set.get_shape()[2].value
        att_fc1 = tf.layers.dense(feature_set, d // 4 , activation=None, use_bias=False, name=name + 'fc1') # (?,?,d // 4)
        att_activation = tf.nn.relu(att_fc1)
        att_fc2 = tf.layers.dense(att_activation, d , activation=None, use_bias=False, name=name + 'fc2') # (?,?,d)
        
        att_scores = tf.sigmoid(att_fc2) # (?,?,d) 别名 tf.nn.sigmoid
        f_agg = feature_set * att_scores # (?,?,d)
        f_agg = helper_tf_util.conv2d(tf.expand_dims(f_agg, axis=2), d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)#(?,?,1,16)
        return f_agg
    
