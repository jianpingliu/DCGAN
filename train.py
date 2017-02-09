import datetime
import os

import numpy as np
import tensorflow as tf

import dcgan

import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './celebA_log_dir', 'log directory')
tf.app.flags.DEFINE_integer('train_steps', 1000, 'number of train steps')


def train():
    # placeholder for z
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name='z')

    # get images
    images = dcgan.inputs(batch_size=FLAGS.batch_size)

    # logits
    D_logits_real, D_logits_fake, generated_images = dcgan.inference(images, z)
    # loss
    d_loss, g_loss = dcgan.loss(D_logits_real, D_logits_fake)

    # train the model
    train_d_op, train_g_op = dcgan.train(d_loss, g_loss)

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        saver = tf.train.Saver()

        training_steps = FLAGS.train_steps

        for step in range(training_steps):

            random_z = np.random.uniform(
                -1, 1, size=(FLAGS.batch_size, FLAGS.z_dim)).astype(np.float32)

            sess.run(train_d_op, feed_dict={z: random_z})
            sess.run(train_g_op, feed_dict={z: random_z})
            sess.run(train_g_op, feed_dict={z: random_z})

            discrimnator_loss, generator_loss = sess.run(
                [d_loss, g_loss], feed_dict={z: random_z})

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, d_loss {:g}, g_loss {:g}".format(
                time_str, step, discrimnator_loss, generator_loss))

            test_images = sess.run(generated_images, feed_dict={z: random_z})

            image_path = os.path.join(FLAGS.log_dir,
                                      "sampled_images_%d.jpg" % step)

            utils.grid_plot(test_images, [8, 8], image_path)

            if step % 100 == 0:
                saver.save(sess, os.path.join(FLAGS.log_dir, "model.ckp"))


def main(argv=None):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    train()


if __name__ == '__main__':
    tf.app.run()
