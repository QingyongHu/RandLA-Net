from os import makedirs
from os.path import exists, join

from tensorflow import data
from helper_ply import read_ply, write_ply
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import confusion_matrix


def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, config, restore_snap=None):
        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(model.logits)
        self.test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
                           for l in dataset.input_trees['test']]

        self.config = config
        self.log_out = open('log_test_' + dataset.name + '.txt', 'a')

    def test(self, model, dataset, num_votes=100, eval=False):

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'predictions')) if not exists(join(test_path, 'predictions')) else None
        # makedirs(join(test_path, 'probs')) if not exists(join(test_path, 'probs')) else None

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5
        t0 = time.time()

        while last_min < num_votes:

            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                    dataset.min_possibility['test'])), self.log_out)

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility['test'])
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)

                if last_min + 1 < new_min:

                    print('Prediction done in {:.1f} s\n'.format(time.time() - t0))
                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # Get file
                        if eval:
                            points, gt = self.load_evaluation_points(file_path)
                        else:
                            points = self.load_test_points(file_path)

                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], 8], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :]

                        # Insert false columns for ignored labels
                        probs2 = probs
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        ply_name = join(test_path, 'predictions', cloud_name)
                        write_ply(ply_name, [points, preds], ['x', 'y', 'z', 'preds'])
                        log_string(ply_name + ' has saved', self.log_out)

                        # evaluate prediction results
                        if eval:
                            self.evaluate(preds, gt)

                        i_test += 1

                    t2 = time.time()
                    print('Reprojection and saving done in {:.1f} s\n'.format(t2 - t1))
                    self.sess.close()
                    return

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return

    @staticmethod
    def load_test_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
    
    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        label = data['scalar_Label'].astype(np.uint8)
        return xyz, label
    
    def evaluate(self, pred, gt):
        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        if not self.config.ignored_label_inds:
            pred_valid = pred
            labels_valid = gt
        else:
            invalid_idx = np.where(gt == self.config.ignored_label_inds)[0]
            labels_valid = np.delete(gt, invalid_idx)
            labels_valid = labels_valid - 1
            pred_valid = np.delete(pred, invalid_idx)
            pred_valid = pred_valid - 1

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
        gt_classes += np.sum(conf_matrix, axis=1)
        positive_classes += np.sum(conf_matrix, axis=0)
        true_positive_classes += np.diagonal(conf_matrix)

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_string('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.log_out)
        log_string('mean IOU:{}'.format(mean_iou), self.log_out)

        mean_iou = 100 * mean_iou
        log_string('Mean IoU = {:.1f}%'.format(mean_iou), self.log_out)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_string('-' * len(s), self.log_out)
        log_string(s, self.log_out)
        log_string('-' * len(s) + '\n', self.log_out)
        return mean_iou


