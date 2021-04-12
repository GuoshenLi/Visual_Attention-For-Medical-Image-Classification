import time

from ops import *
from datetime import timedelta
import numpy as np
import tensorflow as tf
from trainable_image_sampler import get_resampled_images
from dataset_test import data_loader, train_parser, test_parser


class DenseNet:
    def __init__(self):

        self.data_shape = (128, 128, 3)
        self.n_classes = 3
        self.growth_rate = 12
        self.first_output_features = self.growth_rate * 2
        self.total_blocks = 4
        self.layers_per_block = [4, 8, 12, 8]
        self.reduction = 0.5

        self.keep_prob = 1
        self.weight_decay = 0.00001

        self.should_save_logs = True
        self.should_save_model = True
        self.batches_step = 0

        self.num_train = 1449
        self.num_test = 363

        self.src_size = 320
        self.dst_size = 128

        self.margin = 0.05

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self):
        print("Trainable parameters:",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    @property
    def save_path(self):
        save_path = 'previous_ckpt/B2_3TOA_previous/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, 'model.ckpt-270')
        return save_path

    @property
    def logs_path(self):
        logs_path = 'logs'
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        return logs_path



    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        self.saver.restore(self.sess, self.save_path)

        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss1, loss2, loss, TE_loss, accuracy1, accuracy2, accuracy,
                          epoch, prefix, should_print=True):
        if should_print:
            print("mean cross_entropy1: %f, mean accuracy1: %f" % (loss1, accuracy1))
            print('')
            print("mean cross_entropy2: %f, mean accuracy2: %f" % (loss2, accuracy2))
            print('')
            print("mean loss in sum branch: %f, mean accuracy: %f" % (loss, accuracy))
            print('')
            print("mean TE loss: %f" % (TE_loss))

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='loss1_%s' % prefix, simple_value=float(loss1)),
            tf.Summary.Value(tag='loss2_%s' % prefix, simple_value=float(loss2)),
            tf.Summary.Value(tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(tag='accuracy1_%s' % prefix, simple_value=float(accuracy1)),
            tf.Summary.Value(tag='accuracy2_%s' % prefix, simple_value=float(accuracy2)),
            tf.Summary.Value(tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])

        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        # shape = [self.batch_size]
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='images_input1')

        self.high_res_images = tf.placeholder(
            tf.float32,
            # shape=[self.batch_size, self.src_size, self.src_size, 3],
            shape=[None, self.src_size, self.src_size, 3],
            name='high_resolution_input_images')

        self.labels = tf.placeholder(
            tf.float32,
            # shape=[self.batch_size, self.n_classes],
            shape=[None, self.n_classes],
            name='labels')

        self.gt_map = tf.placeholder(
            tf.float32,
            # shape=[self.batch_size, self.dst_size, self.dst_size, 1],
            shape=[None, self.dst_size, self.dst_size, 1],
            name='gt_map')

        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')

        self.batch_size = tf.placeholder(
            tf.int32, shape=[]
        )

        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        # the function batch_norm, conv2d, dropout are defined in the following part.
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)

            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """


        bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
        comp_out = self.composite_function(bottleneck_out, out_features=growth_rate, kernel_size=3)

        output = tf.concat(axis=3, values=(_input, comp_out))
        return output

    def add_block(self, block, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block[block]):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and then average
        pooling
        """
        # call composite function with 1x1 kernel
        # reduce the number of feature maps by compression
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output


    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        spatial_features = output

        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        print(output.shape)
        # FC

        features = tf.layers.flatten(output)

        with tf.variable_scope("final_layer") as scope:
            output = self.conv2d(output, out_features=3, kernel_size=1)

        print(output.shape)
        logits = tf.reshape(output, [-1, self.n_classes])
        print(features.shape, logits.shape)

        return features, logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):

        output = tf.contrib.layers.batch_norm(
            _input, decay=0.9, epsilon=1e-05,
            center=True, scale=True, is_training=self.is_training,
            updates_collections=tf.GraphKeys.UPDATE_OPS)

        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output


    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())


    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def rank_loss(self, pred1, pred2):
        # The ranking loss between two branches,
        # if prob2 > prob1 + margin, rank_loss = 0
        # if prob2 < prob1 + margin, rank_loss = prob1 - prob2 + margin
        prob1 = tf.reduce_sum(tf.multiply(self.labels, pred1), axis=1)  # (batch_size, 1)
        prob2 = tf.reduce_sum(tf.multiply(self.labels, pred2), axis=1)  # (batch_size, 1)
        rank_loss = tf.reduce_mean(tf.maximum(0.0, prob1 - prob2 + self.margin))  # scalar
        return rank_loss

    def hw_flatten(self, x):
        return tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2], x.shape[3]])


    def nonlocal_block(self, x, de=4, scope='Nonlocal', trainable=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            channels = int(x.shape[-1])


            with tf.variable_scope('f'):
              f = self.conv2d(x, out_features = channels // de, kernel_size = 1)
            with tf.variable_scope('g'):
              g = self.conv2d(x, out_features = channels // de, kernel_size = 1)

            with tf.variable_scope('h'):
              h = self.conv2d(x, out_features = channels, kernel_size = 1)


            s = tf.matmul(self.hw_flatten(g), self.hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta_a = tf.nn.softmax(s, dim=-1)  # attention map

            o = tf.matmul(beta_a, self.hw_flatten(h))  # [bs, N, C]

            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=tf.shape(x))  # [bs, h, w, C]

            x = gamma * o + x
            print("Nonlocal output:", x)
        return s, x

    def compute_saliency(self, f_maps, mode="avg"):

        if mode == "avg":
            f_maps = tf.nn.relu(f_maps)
            s_map = tf.reduce_mean(f_maps, axis=-1, keepdims=True)
        elif mode == "max":
            f_maps = tf.nn.relu(f_maps)
            s_map = tf.reduce_max(f_maps, axis=-1, keepdims=True)
        elif mode == "sum_abs":
            f_maps = tf.abs(f_maps)
            s_map = tf.reduce_sum(f_maps, axis=-1, keepdims=True)

        s_map_min = tf.reduce_min(s_map, axis=[1, 2, 3], keepdims=True)  # [batch_size, 8, 8, 1]
        s_map_max = tf.reduce_max(s_map, axis=[1, 2, 3], keepdims=True)
        s_map = tf.div(s_map - s_map_min + 1e-8, s_map_max - s_map_min + 1e-8)  # [batch_size, 8, 8, 1]
        # s_map = tf.div(s_map, s_map_max)

        # s_map = tf.sigmoid(s_map)

        s_map_ = tf.image.resize_images(s_map, size=(31, 31))  # used for image resampling.

        saliency_map = tf.tile(s_map, (1, 1, 1, 3))
        saliency_map = tf.image.resize_images(saliency_map, (128, 128))  # (8, 128, 128, 3)

        return s_map_, saliency_map

    def select_samples(self, pred1, pred2, mode="net2_teach_net1"):
        """
        Select the samples that pred2 are more accurate than pred1,
        and also, pred2 make correct decision,
        then let the resampled att_maps1 of these samples be similar with att_maps2.
        """
        prob1 = tf.reduce_sum(self.labels * pred1, axis=1)
        prob2 = tf.reduce_sum(self.labels * pred2, axis=1)
        threshold = 0.5

        cond1 = tf.cast(tf.greater(prob2, tf.maximum(prob1, threshold)), tf.float32)
        cond2 = tf.cast(tf.equal(tf.argmax(pred2, 1), tf.argmax(self.labels, 1)), tf.float32)
        cond3 = tf.cast(tf.greater(tf.argmax(self.labels, 1), 0), tf.float32)

        if mode == "net2_teach_net1":
            select = cond3
        elif mode == "net1_teach_net2":
            select = tf.multiply(tf.multiply(cond1, cond2), cond3)

        return select


    def network(self, _input):
        """
        Define the structure of the backbone network,
        this network will be used in both two branches.
        Input: image,
        Return: the feature vector of input image(batch_size, channels),
        predicted logits(batch_size, 3), predicted spatial probability(batch_size, h, w, 3).
        """
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block

        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                _input,
                out_features=self.first_output_features,
                kernel_size=3, strides=[1, 1, 1, 1])
            print(output.shape)

        with tf.variable_scope("Initial_pooling"):
            output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(output.shape)
            initial_pool = output

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(block, output, growth_rate, layers_per_block)
            print(output.shape)

            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)
                    print(output.shape)

            if block == 0:
                cor1, output = self.nonlocal_block(output, de=4, scope='NLB1')
                fmaps_b1 = output
            if block == 1:
                cor2, output = self.nonlocal_block(output, de=4, scope='NLB2')
                fmaps_b2 = output

            if block == 2:
                cor3, output = self.nonlocal_block(output, de=4, scope='NLB3')
                fmaps_b3 = output

        f_maps = output


        with tf.variable_scope("Transition_to_classes"):
            features, logits = self.transition_layer_to_classes(f_maps)

        return f_maps, fmaps_b1, fmaps_b2, fmaps_b3, features, logits


    def network_merge_net1_fmap(self, _input, fmap2_b1_concat, fmap2_b2_concat, fmap2_b3_concat):
        """
        Define the structure of the backbone network,
        this network will be used in both two branches.
        Input: image,
        Return: the feature vector of input image(batch_size, channels),
        predicted logits(batch_size, 3), predicted spatial probability(batch_size, h, w, 3).
        """
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block

        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                _input,
                out_features=self.first_output_features,
                kernel_size=3, strides=[1, 1, 1, 1])
            print(output.shape)

        with tf.variable_scope("Initial_pooling"):
            output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            print(output.shape)
            initial_pool = output

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(block, output, growth_rate, layers_per_block)
            print(output.shape)

            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)
                    print(output.shape)

            if block == 0:
                cor1, output = self.nonlocal_block(output, de=4, scope='NLB1')
                fmaps_b1 = output
                output = tf.concat((output, fmap2_b1_concat), axis = -1)
            if block == 1:
                cor2, output = self.nonlocal_block(output, de=4, scope='NLB2')
                fmaps_b2 = output
                output = tf.concat((output, fmap2_b2_concat), axis = -1)
            if block == 2:
                cor3, output = self.nonlocal_block(output, de=4, scope='NLB3')
                fmaps_b3 = output
                output = tf.concat((output, fmap2_b3_concat), axis = -1)

        f_maps = output

        # the last block is followed by a "transition_to_classes" layer.
        with tf.variable_scope("Transition_to_classes"):
            features, logits = self.transition_layer_to_classes(f_maps)

        return f_maps, fmaps_b1, fmaps_b2, fmaps_b3, features, logits



    def _build_graph(self):
        with tf.variable_scope("net1") as scope:
            f_maps1, fmaps1_b1, fmaps1_b2, fmaps1_b3, self.features1, logits1 = self.network(self.images)
            self.pred1 = tf.nn.softmax(logits1)

            self.s_map1, self.saliency1 = self.compute_saliency(f_maps1)

            _, self.smaps1_b1 = self.compute_saliency(fmaps1_b1)
            _, self.smaps1_b2 = self.compute_saliency(fmaps1_b2)
            _, self.smaps1_b3 = self.compute_saliency(fmaps1_b3)

        self.input2 = get_resampled_images(
            self.high_res_images, self.s_map1, self.batch_size, self.src_size, self.dst_size, padding_size=30)


        self.fmaps2_b1_concat = get_resampled_images(
            fmaps1_b1, self.s_map1, self.batch_size, self.dst_size // 4, self.dst_size // 4, padding_size=30)
        _, self.fmaps2_b1_plot = self.compute_saliency(self.fmaps2_b1_concat)

        self.fmaps2_b2_concat = get_resampled_images(
            fmaps1_b2, self.s_map1, self.batch_size, self.dst_size // 8, self.dst_size // 8, padding_size=30)
        _, self.fmaps2_b2_plot = self.compute_saliency(self.fmaps2_b2_concat)

        self.fmaps2_b3_concat = get_resampled_images(
            fmaps1_b3, self.s_map1, self.batch_size, self.dst_size // 16, self.dst_size // 16, padding_size=30)
        _, self.fmaps2_b3_plot = self.compute_saliency(self.fmaps2_b3_concat)

        self.resampled_s_map1 = get_resampled_images(self.s_map1, self.s_map1, self.batch_size, 31, 8, padding_size=30)
        print(self.resampled_s_map1.shape)

        self.resampled_gt = get_resampled_images(
            self.gt_map, self.s_map1, self.batch_size, self.dst_size, self.dst_size, padding_size=30)

        resampled_saliency1 = tf.tile(self.resampled_s_map1, (1, 1, 1, 3))
        self.resampled_saliency1 = tf.image.resize_images(resampled_saliency1, (128, 128))  # (8, 128, 128, 3)


        with tf.variable_scope("net2") as scope:
            # f_maps2, fmaps2_b1, fmaps2_b2, fmaps2_b3, self.features2, logits2, spatial_pred2 = self.network(self.input2)
            f_maps2, fmaps2_b1, fmaps2_b2, fmaps2_b3, self.features2, logits2 = self.network_merge_net1_fmap(self.input2, self.fmaps2_b1_concat, self.fmaps2_b2_concat, self.fmaps2_b3_concat)
            self.pred2 = tf.nn.softmax(logits2)

            _, self.smaps2_b1 = self.compute_saliency(fmaps2_b1)
            _, self.smaps2_b2 = self.compute_saliency(fmaps2_b2)
            _, self.smaps2_b3 = self.compute_saliency(fmaps2_b3)


            # Saliency of the original features of block4 and the SACA feature.
            self.s_map2, self.saliency2 = self.compute_saliency(f_maps2)
            self.s_map2 = tf.image.resize_images(self.s_map2, [8, 8])  # used for TE-loss computation.

        ##########################################################
        # The 3-rd branch: combine the features from net1 and net2,
        # and, then make predictions of the combined features.
        ##########################################################

        total_fmaps = tf.concat(axis=3, values=(f_maps1, f_maps2))

        with tf.variable_scope("Sum_branch") as scope:
            _, logits = self.transition_layer_to_classes(total_fmaps)
            self.pred = tf.nn.softmax(logits)

        # weighted loss:
        class_weights = tf.constant([1, 1, 1])
        weights = tf.gather(class_weights, tf.argmax(self.labels, axis=-1))
        cross_entropy1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits1, weights=weights, label_smoothing=0.1))
        self.cross_entropy1 = cross_entropy1

        self.kd_loss1 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.pred, logits1, weights=weights, label_smoothing=0.1))

        cross_entropy2 = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits2, weights=weights, label_smoothing=0.1))
        self.cross_entropy2 = cross_entropy2

        self.sum_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            self.labels, logits, weights=weights, label_smoothing=0.1))

        self.rank_loss_2_1 = self.rank_loss(self.pred1, self.pred2)  # force the pred2 more accurate than pred1

        self.TE_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
            tf.square(self.resampled_s_map1 - self.s_map2), axis=[1, 2, 3])))


        var_list = [var for var in tf.trainable_variables()]
        var_list1 = [var for var in tf.trainable_variables() if var.name.split('/')[0] == 'net1']
        var_list2 = [var for var in tf.trainable_variables() if var.name.split('/')[0] == 'net2']
        var_list3 = [var for var in tf.trainable_variables() if var.name.split('_')[0] == 'Sum']


        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        l2_loss1 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('/')[0] == 'net1'])
        l2_loss2 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('/')[0] == 'net2'])
        l2_loss3 = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables() if var.name.split('_')[0] == 'Sum'])


        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        update_ops_1 = [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'net1' in ops.name]
        with tf.control_dependencies(update_ops_1):
            self.train_step1 = optimizer.minimize(
                self.cross_entropy1 + l2_loss1 * self.weight_decay, var_list=var_list1)

        update_ops_2 = [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'net2' in ops.name]
        with tf.control_dependencies(update_ops_2):
            self.train_step2 = optimizer.minimize(
                self.cross_entropy2 + l2_loss2 * self.weight_decay, var_list=var_list2)

        update_ops_sum = [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'Sum' in ops.name]
        with tf.control_dependencies(update_ops_sum):
            self.train_step3 = optimizer.minimize(
                self.sum_loss + l2_loss3 * self.weight_decay, var_list=var_list3)


        correct_prediction1 = tf.equal(
            tf.argmax(self.pred1, 1),
            tf.argmax(self.labels, 1))
        self.correct_prediction1 = correct_prediction1
        self.accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

        correct_prediction2 = tf.equal(
            tf.argmax(self.pred2, 1),
            tf.argmax(self.labels, 1))
        self.correct_prediction2 = correct_prediction2
        self.accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

        self.correct_prediction = tf.equal(
            tf.argmax(self.pred, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        # reduce the lr at epoch1 and epoch2.
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']


        total_start_time = time.time()

        loss1_all_epochs = []
        loss2_all_epochs = []
        loss3_all_epochs = []

        acc1_all_epochs = []
        acc2_all_epochs = []
        acc_all_epochs = []

        nr1_all_epochs = []
        br1_all_epochs = []
        ir1_all_epochs = []
        kappa1_all_epochs = []

        nr2_all_epochs = []
        br2_all_epochs = []
        ir2_all_epochs = []
        kappa2_all_epochs = []

        nr_all_epochs = []
        br_all_epochs = []
        ir_all_epochs = []
        kappa_all_epochs = []

        """
        Only save the model with highest accuracy2
        """

        best_acc = 0.0

        self.train_high_image_batch, self.train_low_image_batch, self.train_label_batch = data_loader(
            tfrecord_path='drive/MyDrive/Google_Deep_Learning/WCE_data_warped/tfrecord/train3_zoom',
            parser=train_parser, batch_size=batch_size, epoch=n_epochs, type='train')
        self.test_high_image_batch, self.test_low_image_batch, self.test_label_batch = data_loader(
            tfrecord_path='drive/MyDrive/Google_Deep_Learning/WCE_data_warped/tfrecord/test3_zoom', parser=test_parser,
            batch_size=batch_size, epoch=n_epochs, type='test')

        for epoch in range(1, n_epochs + 1):
            self.epoch = epoch
            start_time = time.time()

            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')

            if epoch == reduce_lr_epoch_1:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)

            if epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 2
                print("Decrease learning rate, new lr = %f" % learning_rate)

            print("Training...")
            start_train = time.time()
            loss1, loss2, loss3, TE_loss, rank_loss, acc1, acc2 = self.train_one_epoch(batch_size, learning_rate)
            train_time = time.time() - start_train
            total_train_time = int(n_epochs * train_time)
            print("Training time per epoch: %s, Total training time: %s" % (
                str(timedelta(seconds=train_time)),
                str(timedelta(seconds=total_train_time))))

            if self.should_save_logs:
                self.log_loss_accuracy(loss1, loss2, loss3, TE_loss, acc1, acc2, acc2, epoch, prefix='train')

            print("Rank loss:", rank_loss)

            loss1_all_epochs.append(loss1)
            loss2_all_epochs.append(loss2)
            loss3_all_epochs.append(loss3)

            if train_params.get('validation_set'):
                print("Validation...")
                start_val = time.time()
                loss1, loss2, loss3, TE_loss, acc1, acc2, acc, nr1, br1, ir1, kappa1, \
                nr2, br2, ir2, kappa2, nr, br, ir, kappa = self.test(batch_size)
                val_time_per_image = (time.time() - start_val) / self.num_test
                print("Validation time per image:", str(timedelta(seconds=val_time_per_image)))

                if self.should_save_logs:
                    self.log_loss_accuracy(loss1, loss2, loss3, TE_loss, acc1, acc2, acc, epoch, prefix='valid')

                acc1_all_epochs.append(acc1)
                acc2_all_epochs.append(acc2)
                acc_all_epochs.append(acc)

                nr1_all_epochs.append(nr1)
                br1_all_epochs.append(br1)
                ir1_all_epochs.append(ir1)
                kappa1_all_epochs.append(kappa1)

                nr2_all_epochs.append(nr2)
                br2_all_epochs.append(br2)
                ir2_all_epochs.append(ir2)
                kappa2_all_epochs.append(kappa2)

                nr_all_epochs.append(nr)
                br_all_epochs.append(br)
                ir_all_epochs.append(ir)
                kappa_all_epochs.append(kappa)

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                if epoch >= 200 and epoch % 5 == 0:  #
                    self.save_model(global_step=epoch)

                if acc >= best_acc:
                    best_acc = acc
                    self.save_model(global_step=1)



        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))


    def train_one_epoch(self, batch_size, learning_rate):



        total_loss1 = []
        total_loss2 = []
        total_loss3 = []
        total_TE_loss = []
        total_rank_loss = []

        total_prob1 = []  # the probability of ground truth label class of each image generated by net1
        total_prob2 = []  # the probability of ground truth label class of each image generated by net2

        total_pred1 = []  # the predicted class label of each image by net1
        total_pred2 = []  # the predicted class label of each image by net2
        total_labels1 = []  # the ground truth label of each image

        for i in range(int(np.ceil(self.num_train / batch_size))):

            ########## adjust the batch_size of last batch ############

            high_images, low_images, labels = self.sess.run([
                self.train_high_image_batch, self.train_low_image_batch, self.train_label_batch])

            if (self.num_train % batch_size == 0) or (i != int(np.ceil(self.num_train / batch_size)) - 1):
                feed_dict = {
                    self.images: low_images,
                    self.high_res_images: high_images,
                    self.labels: labels,
                    self.learning_rate: learning_rate,
                    self.is_training: True,
                    self.batch_size: batch_size
                }

            else:

                feed_dict = {
                    self.images: low_images,
                    self.high_res_images: high_images,
                    self.labels: labels,
                    self.learning_rate: learning_rate,
                    self.is_training: True,
                    self.batch_size: self.num_train - batch_size * int(np.floor(self.num_train / batch_size))
                }


            fetches = [self.train_step1, self.train_step2, self.train_step3, self.features1, self.features2,
                       self.cross_entropy1, self.cross_entropy2, self.sum_loss, self.TE_loss, self.rank_loss_2_1,
                       self.accuracy1, self.accuracy2, self.pred1, self.pred2]

            results = self.sess.run(fetches, feed_dict=feed_dict)
            _, _, _, features1, features2, loss1, loss2, loss3, TE_loss, rank_loss, acc1, acc2, pred1, pred2 = results

            total_loss1.append(loss1)
            total_loss2.append(loss2)
            total_loss3.append(loss3)
            total_TE_loss.append(TE_loss)
            total_rank_loss.append(rank_loss)

            prob1 = np.sum(np.multiply(pred1, labels), axis=1)  # [batch_size,]
            prob2 = np.sum(np.multiply(pred2, labels), axis=1)

            total_prob1.extend(prob1)
            total_prob2.extend(prob2)

            total_pred1.extend(np.argmax(pred1, axis=1))  # the predicted class label that output by net 1
            total_pred2.extend(np.argmax(pred2, axis=1))  # the predicted class label that output by net 2
            total_labels1.extend(np.argmax(labels, axis=1))  # the ground truth label of the data in the batch

            if self.should_save_logs:
                self.batches_step += 1
                # save loss and accuracy into Summary
                self.log_loss_accuracy(
                    loss1, loss2, loss3, TE_loss, acc1, acc2, acc2, self.batches_step,
                    prefix='per_batch', should_print=False)

        mean_loss1 = np.mean(total_loss1)
        mean_loss2 = np.mean(total_loss2)
        mean_loss3 = np.mean(total_loss3)
        mean_TE_loss = np.mean(total_TE_loss)
        mean_rank_loss = np.mean(total_rank_loss)

        overall_acc_1, normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1 = get_accuracy(
            preds=total_pred1, labels=total_labels1)
        overall_acc_2, normal_recall_2, bleed_recall_2, inflam_recall_2, kappa_2 = get_accuracy(
            preds=total_pred2, labels=total_labels1)


        return mean_loss1, mean_loss2, mean_loss3, mean_TE_loss, \
               mean_rank_loss, overall_acc_1, overall_acc_2

    def test(self, batch_size):
        saliency_path = "./visualize_heatmaps_test/attention_all_guided"

        if not os.path.exists(saliency_path):
            os.makedirs(saliency_path)

        total_loss1 = []
        total_loss2 = []
        total_loss3 = []
        total_TE_loss = []

        total_prob1 = []
        total_prob2 = []

        total_pred1 = []
        total_pred2 = []
        total_pred = []
        total_labels = []

        self.test_high_image_batch,  self.test_low_image_batch,  self.test_label_batch  = data_loader(tfrecord_path = 'tfrecord/test',  parser = test_parser,  batch_size = batch_size, epoch = 1, type = 'test')

        for i in range(int(np.ceil(self.num_test / batch_size))):

            ########## adjust the batch_size of last batch ############

            test_high_images, test_low_images, test_labels = self.sess.run([
                self.test_high_image_batch, self.test_low_image_batch, self.test_label_batch])

            if (self.num_test % batch_size == 0) or (i != int(np.ceil(self.num_test / batch_size)) - 1):
                feed_dict = {
                    self.images: test_low_images,
                    self.high_res_images: test_high_images,
                    self.labels: test_labels,
                    self.is_training: False,
                    self.batch_size: batch_size
                }

            else:

                feed_dict = {
                    self.images: test_low_images,
                    self.high_res_images: test_high_images,
                    self.labels: test_labels,
                    self.is_training: False,
                    self.batch_size: self.num_test - batch_size * int(np.floor(self.num_test / batch_size))
                }


            fetches = [self.input2, self.cross_entropy1, self.cross_entropy2, self.sum_loss, self.TE_loss, \
                       self.saliency1, self.saliency2,\
                       self.smaps1_b1, self.smaps1_b2, self.smaps1_b3, \
                       self.fmaps2_b1_plot, self.fmaps2_b2_plot, self.fmaps2_b3_plot, \
                       self.smaps2_b1, self.smaps2_b2, self.smaps2_b3, \
                       self.pred1, self.pred2, self.pred]


            input2, loss1, loss2, loss3, TE_loss, s_map1, s_map2, smap1_b1, smap1_b2, smap1_b3, smap1_b1_zoom, smap1_b2_zoom, smap1_b3_zoom, smap2_b1, smap2_b2, smap2_b3, pred1, pred2, pred = \
                self.sess.run(fetches, feed_dict=feed_dict)

            total_loss1.append(loss1)
            total_loss2.append(loss2)
            total_loss3.append(loss3)
            total_TE_loss.append(TE_loss)

            prob1 = np.sum(np.multiply(pred1, test_labels), axis=1)  # [batch_size]
            prob2 = np.sum(np.multiply(pred2, test_labels), axis=1)

            total_prob1.extend(prob1)
            total_prob2.extend(prob2)

            pred1 = np.argmax(pred1, axis=1)
            pred2 = np.argmax(pred2, axis=1)
            pred = np.argmax(pred, axis=1)
            labels = np.argmax(test_labels, axis=1)

            total_pred1.extend(pred1)
            total_pred2.extend(pred2)
            total_pred.extend(pred)
            total_labels.extend(labels)


            try: # the last batch will out of range, so use try-exept
                for index in range(batch_size):
                    img_index = i * batch_size + index

                    save_img(smap1_b1[index]      , img_index, saliency_path, img_name='_mid1.jpg'       , mode = "heatmap")
                    save_img(smap1_b1_zoom[index] , img_index, saliency_path, img_name='_mid1_zoom.jpg'  , mode = "heatmap")
                    save_img(smap1_b2[index]      , img_index, saliency_path, img_name = '_mid2.jpg'     ,  mode = "heatmap")
                    save_img(smap1_b2_zoom[index] , img_index, saliency_path, img_name = '_mid2_zoom.jpg',  mode = "heatmap")
                    save_img(smap1_b3[index]      , img_index, saliency_path, img_name = '_mid3.jpg'     ,  mode = "heatmap")
                    save_img(smap1_b3_zoom[index] , img_index, saliency_path, img_name = '_mid3_zoom.jpg',  mode = "heatmap")

                    # save_img(smap2_b1[index] , img_index, saliency_path, img_name = '_branch2_mid1.jpg',  mode = "heatmap")
                    # save_img(smap2_b2[index] , img_index, saliency_path, img_name = '_branch2_mid2.jpg',  mode = "heatmap")
                    # save_img(smap2_b3[index] , img_index, saliency_path, img_name = '_branch2_mid3.jpg',  mode = "heatmap")

                    save_img(s_map1[index],  img_index, saliency_path, img_name='_att1.jpg', mode = "heatmap")
                    save_img(s_map2[index], img_index, saliency_path, img_name='_att2.jpg', mode = "heatmap")
                    save_img(test_low_images[index], img_index, saliency_path, img_name = '_input1.jpg',  mode = "image")
                    save_img(input2[index], img_index, saliency_path, img_name = '_input2.jpg', mode = "image")

            except:
                print('save_all_image_and saliency map of one test epoch!')


        mean_loss1 = np.mean(total_loss1)
        mean_loss2 = np.mean(total_loss2)
        mean_loss3 = np.mean(total_loss3)
        mean_TE_loss = np.mean(total_TE_loss)

        overall_acc_1, normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1 = get_accuracy(preds=total_pred1,
                                                                                                labels=total_labels)
        overall_acc_2, normal_recall_2, bleed_recall_2, inflam_recall_2, kappa_2 = get_accuracy(preds=total_pred2,
                                                                                                labels=total_labels)
        overall_acc, normal_recall, bleed_recall, inflam_recall, kappa = get_accuracy(preds=total_pred,
                                                                                      labels=total_labels)




        return mean_loss1, mean_loss2, mean_loss3, mean_TE_loss, overall_acc_1, overall_acc_2, overall_acc, \
               normal_recall_1, bleed_recall_1, inflam_recall_1, kappa_1, normal_recall_2, bleed_recall_2, \
               inflam_recall_2, kappa_2, normal_recall, bleed_recall, inflam_recall, kappa