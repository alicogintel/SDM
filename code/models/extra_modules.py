import tensorflow as tf
from tensorflow.python.framework import dtypes


class ExtraModules:

    def __init__(self, parser, hparams):
        pass

    def average_item_embedding(self, inputs):
        """
        :param inputs: [batch_size, time, units]
        :return: masked average pooling
        """
        with tf.variable_scope('item_avg_pooling', partitioner=self.partitioner):
            length = tf.reshape(tf.shape(inputs)[1], [-1])
            # [time, time]
            lower_tri = tf.ones(tf.concat([length, length], axis=0))
            # [time, time]
            lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()
            # [batch_size, time, time]
            masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(inputs)[0], 1, 1])
            # [batch_size, time, units]
            output = tf.matmul(masks, inputs)
            # [time]
            avg_num = tf.range(1, 1 + tf.shape(inputs)[1])
            avg_num = tf.cast(avg_num, dtypes.float32)
            # [1, time, 1]
            avg_num = tf.reshape(avg_num, [1, tf.shape(avg_num)[0], 1])
            # [batch_size, time, 1]
            avg_num = tf.tile(avg_num, [tf.shape(inputs)[0], 1, 1])
            # [batch_size, time, units]
            output = tf.divide(output, avg_num)
        return output

    def create_dnn(self, item_embedding, prefer_embedding, user_embedding):
        """
        :param prefer_embedding:
        :param item_embedding:
        :param user_embedding:
        :return:
        """
        with tf.variable_scope('dnn', partitioner=self.partitioner):
            user_embedding = tf.tile(tf.expand_dims(user_embedding, 1), [1, tf.shape(item_embedding)[1], 1])
            prefer_embedding = tf.tile(tf.expand_dims(prefer_embedding, 1), [1, tf.shape(item_embedding)[1], 1])
            output = tf.concat([item_embedding, prefer_embedding, user_embedding], -1)
            output = tf.layers.dropout(output, self.dropout, training=self.is_training)
            output = tf.layers.dense(output, 4 * self.num_units, tf.nn.relu)
            output = tf.layers.dropout(output, self.dropout, training=self.is_training)
            output = tf.layers.dense(output, 2 * self.num_units, tf.nn.relu)
            output = tf.layers.dropout(output, self.dropout, training=self.is_training)
            output = tf.layers.dense(output, self.num_units, tf.nn.relu)
        return output

    def create_in_shan(self, item_embedding, prefer_embedding):
        """
        :param item_embedding:
        :param user_embedding:
        :param prefer_embedding:
        :param num_units:
        :return:
        """
        with tf.variable_scope('shan_in', partitioner=self.partitioner):
            prefer_embedding = tf.reshape(prefer_embedding, [-1, 1, tf.shape(prefer_embedding)[-1]])
            output = tf.concat([prefer_embedding, item_embedding], 1)
        return output

    def create_out_shan(self, inputs):
        """
        :param inputs:
        :return:
        """
        with tf.variable_scope('shan_out', partitioner=self.partitioner):
            output = tf.slice(inputs, [0, 1, 0], [tf.shape(inputs)[0], tf.shape(inputs)[1] - 1, tf.shape(inputs)[2]])
        return output
