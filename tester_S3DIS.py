from os import makedirs
from os.path import exists, join
from helper_ply import write_ply
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time


def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open('log_test_' + str(dataset.val_split) + '.txt', 'a')

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['validation']]

    def test(self, model, dataset, num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],
                       )

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)
                acc = correct / float(np.prod(np.shape(stacked_labels)))
                print('step' + str(step_id) + ' acc:' + str(acc))
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                step_id += 1

            except tf.errors.OutOfRangeError:

                new_min = np.min(dataset.min_possibility['validation'])
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    log_out('\nConfusion on sub clouds', self.Log_file)
                    confusion_list = []

                    num_val = len(dataset.input_labels['validation'])

                    for i_test in range(num_val):
                        probs = self.test_probs[i_test]
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        labels = dataset.input_labels['validation'][i_test]

                        # Confs
                        confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    log_out(s + '\n', self.Log_file)

                    if int(np.ceil(new_min)) % 1 == 0:

                        # Project predictions
                        log_out('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), self.Log_file)
                        proj_probs_list = []

                        for i_val in range(num_val):
                            # Reproject probs back to the evaluations points
                            proj_idx = dataset.val_proj[i_val]
                            probs = self.test_probs[i_val][proj_idx, :]
                            proj_probs_list += [probs]

                        # Show vote results
                        log_out('Confusion on full clouds', self.Log_file)
                        confusion_list = []
                        for i_test in range(num_val):
                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)

                            # Confusion
                            labels = dataset.val_labels[i_test]
                            acc = np.sum(preds == labels) / len(labels)
                            log_out(dataset.input_names['validation'][i_test] + ' Acc:' + str(acc), self.Log_file)

                            confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]
                            name = dataset.input_names['validation'][i_test] + '.ply'
                            write_ply(join(test_path, 'val_preds', name), [preds, labels], ['pred', 'label'])

                        # Regroup confusions
                        C = np.sum(np.stack(confusion_list), axis=0)

                        IoUs = DP.IoU_from_confusions(C)
                        m_IoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * m_IoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        log_out('-' * len(s), self.Log_file)
                        log_out(s, self.Log_file)
                        log_out('-' * len(s) + '\n', self.Log_file)
                        print('finished \n')
                        self.sess.close()
                        return

                self.sess.run(dataset.val_init_op)
                epoch_id += 1
                step_id = 0
                continue

        return
