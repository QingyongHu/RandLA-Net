from os.path import join
from RandLANet import Network
from tester_S3DIS import ModelTester
from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os


class S3DIS:
    def __init__(self, test_area_idx):
        self.name = 'S3DIS'
        self.path = '/home/rose/projects/RandLA-Net/data/S3DIS'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))

        # Initiate containers 初始化列表和字典来保存输入
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):# 加载采样点云子集
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7 
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices 验证和测试时转换为原始输入Index
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    # Generate the input data flow 每个batch生成输入数据流
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize   随机初始化概率
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]    # 点
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))] # 点云

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability 选择概率最小的点云
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point 选择云中可能性最小的点作为查询点
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure 从树结构中获取点云中的所有点
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region 输入区域点的中心
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point 对中心点加噪声
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                # 检查所选云中的点数是否小于预定义的点数
                if len(points) < cfg.num_points:
                    # Query all points within the cloud 选择点云中所有的点
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points 选择预设的点数
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0] # 预设40960

                # Shuffle index 打乱点顺序
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index 基于index获得相应的点坐标和颜色
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point # 相对坐标
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx] # 颜色
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx] # 标签

                # Update the possibility of the selected points # 更新选择点的概率 delta = (1 - dists/max(dists))^2
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement 如果选择的点少于预设的点数，上采样，随机选择之前的点再Concatenate
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2(): # 将输入展开映射到每一层
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):# 默认5
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32) # 选择k近邻 16， 输入（B*N1*3, B*N2*3, k)，输出B*N2*k K为index
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]  # sub_sampling_ratio = [4, 4, 4, 4, 2] 整数除法
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]  # k近邻对应原始点云下采样
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32) #tf.py_func()接收的是tensor，然后将其转化为numpy array送入我们自定义的my_func函数，最后再将my_func函数输出的numpy array转化为tensor返回。注：已知待采样和上采样点的坐标
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points # 原始点云更新为下采样后的点云

            input_list = input_points + input_neighbors + input_pools + input_up_samples # 原始点云，原始点云k近邻，k近邻对应原始点云下采样，原始点云下采样后上采样
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx] # 再把原始的特征等加在后边

            return input_list

        return tf_map

    def init_input_pipeline(self): # 初始化输入流程
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes) # 训练数据转为tf tensor
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes) # 测试数据转为tf tensor

        self.batch_train_data = self.train_data.batch(cfg.batch_size) # 划分batch
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2() 

        self.batch_train_data = self.batch_train_data.map(map_func=map_func) # 将batch展开到每一层
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        #向工作流的末尾添加 tf.data.Dataset.prefetch(buffer_size)，这样操作能确保下个批次始终对 GPU 可用，减少 GPU 的闲置时间。Buffer_size 是 GPU 预读取的批次数量。通常 buffer_size=1 在多数情况下就足够了，特别是每批次的处理时间不断变化时，这样设置能协调时间。
        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode
    
    start_data = time.time()# 加载数据
    test_area = FLAGS.test_area
    dataset = S3DIS(test_area)
    dataset.init_input_pipeline()
    
    print('加载数据时间',time.time() - start_data)# 加载数据时间
    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        start_net = time.time()# 构建网络
        #import pdb
        #pdb.set_trace() #断点
        model = Network(dataset, cfg)
        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        print('构建网络时间',time.time() - start_net)#构建网络时间
        start_model = time.time()
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        print('模型加载时间',time.time() - start_model)#模型加载时间
        start_test = time.time()
        tester.test(model, dataset)
        print('总共模型测试时间',time.time() - start_test)#模型测试时间
    else:
        ##################
        # Visualize data #
        ##################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :], cfg.num_classes + 1)
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]], cfg.num_classes + 1)
