import tensorflow as tf


def conv_2d(x, num_filters, kernel_size=5, stride=2, scope='conv'):

    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w', [kernel_size, kernel_size, x.get_shape()[-1], num_filters],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        conv = tf.nn.conv2d(
            x, w, strides=[1, stride, stride, 1], padding='SAME')

        biases = tf.get_variable(
            'biases', [num_filters], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_transpose(x,
                     output_shape,
                     kernel_size=5,
                     stride=2,
                     scope="conv_transpose"):

    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w',
            [kernel_size, kernel_size, output_shape[-1], x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        conv_transpose = tf.nn.conv2d_transpose(
            x, w, output_shape, strides=[1, stride, stride, 1])

        biases = tf.get_variable(
            'biases', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))

        conv_transpose = tf.nn.bias_add(conv_transpose, biases)

        return conv_transpose


def fc_2(x,
         num_outputs=1,
         weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
         biases_initializer=tf.constant_initializer(value=0.0),
         reuse=False,
         scope="fc"):

    output = tf.contrib.layers.fully_connected(
        x,
        num_outputs=num_outputs,
        weights_initializer=weights_initializer,
        biases_initializer=biases_initializer,
        reuse=reuse,
        scope=scope)
    return output


def fc(x, num_outputs, scope="fc"):

    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w', [x.get_shape()[-1], num_outputs],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        biases = tf.get_variable(
            'biases', [num_outputs], initializer=tf.constant_initializer(0.0))

        output = tf.nn.bias_add(tf.matmul(x, w), biases)

        return output


def batch_norm(x,
               decay=0.9,
               epsilon=1e-5,
               scale=True,
               is_training=True,
               reuse=False,
               scope='batch_norm'):

    bn = tf.contrib.layers.batch_norm(
        x,
        decay=decay,
        updates_collections=None,
        epsilon=epsilon,
        scale=scale,
        is_training=is_training,
        reuse=reuse,
        scope=scope)
    return bn


def leaky_relu(x, leak=0.2):
    return tf.maximum(x, leak * x)
