"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5
    end_point = ''
    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,keep_prob=dropout_keep_prob)):
            end_point = 'Conv2d_1a_7x7_'
            ##########################
            # Put your code here.
            net = slim.conv2d(images, 112, [7,7], stride = 2, padding ='SAME',  scope=end_point)
            #scope + '_7x7'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            # first block:
            end_point = 'block1'
            net = block(net, layers = 6, growth=8, scope=end_point)
            end_points[end_point] = net

            # transition layer:
            end_point = 'transition1'
            net = slim.conv2d(net, 56, [1, 1], scope=scope)
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
            end_points[end_point] = net

            # second block:
            end_point = 'block1'
            net = block(net, layers=12, growth=8, scope=end_point)
            end_points[end_point] = net

            # transition layer2:
            end_point = 'transition2'
            net = slim.conv2d(net, 28, [1, 1], scope=scope)
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
            end_points[end_point] = net

            # 3rd block:
            end_point = 'block3'
            net = block(net, layers=24, growth=8, scope=end_point)
            end_points[end_point] = net

            # transition layer3:
            end_point = 'transition2'
            net = slim.conv2d(net, 14 , [1, 1], scope=scope)
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope=end_point)
            end_points[end_point] = net

            # 4rd block:
            end_point = 'block4'
            net = block(net, layers=16, growth=8, scope=end_point)
            end_points[end_point] = net


            end_point = 'end'
            net = slim.avg_pool2d(net, [7, 7], scope=end_point)
            end_points[end_point] = net

            logits = tf.squeeze(net, name='SpatialSqueeze')
            #logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            ##########################
        return logits, end_points

def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
