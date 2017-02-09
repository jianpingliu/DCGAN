""" Routine for decoding TFRecords file format """

import tensorflow as tf
import numpy as np


class TFRecordsReader(object):
    def __init__(self,
                 image_height=218,
                 image_width=178,
                 image_channels=3,
                 directory="data",
                 filename_pattern=".tfrecord",
                 crop=False,
                 crop_height=None,
                 crop_width=None,
                 resize=False,
                 resize_height=None,
                 resize_width=None,
                 image_format="jpeg",
                 num_examples_per_epoch=64):
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.directory = directory
        self.filename_pattern = filename_pattern
        self.crop = crop
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.resize = resize
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.image_format = image_format
        self.num_examples_per_epoch = num_examples_per_epoch

    def read_example(self, filename_queue):
        # TFRecoard reader
        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)

        # read data from serialized examples
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)
            })
        label = features['label']
        image = features['image_raw']

        # decode raw image data as integers
        if self.image_format == 'jpeg':
            decoded_image = tf.image.decode_jpeg(
                image, channels=self.image_channels)
        else:
            decoded_image = tf.decode_raw(image, tf.uint8)

        return decoded_image, label

    def _generate_image_and_label_batch(self, image, label, min_queue_examples,
                                        batch_size, shuffle):
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.
        num_preprocess_threads = 1
        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 2 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 2 * batch_size)

        return images, tf.reshape(label_batch, [batch_size])

    def inputs(self, batch_size=64):
        pattern = '%s/%s' % (self.directory, self.filename_pattern)
        filenames = tf.gfile.Glob(pattern)

        # create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # read examples from files in the filename queue
        image, label = self.read_example(filename_queue)

        # image shape
        height = self.image_height
        width = self.image_width
        channels = self.image_channels

        # reshape image tensor
        image = tf.reshape(image, [height, width, channels])

        # crop image
        if self.crop:
            assert isinstance(self.crop_height, int)
            assert isinstance(self.crop_width, int)

            image = tf.image.resize_image_with_crop_or_pad(
                image, self.crop_height, self.crop_width)

        if self.resize:
            size = [self.resize_height, self.resize_width]
            image = tf.image.resize_images(image, size)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        num_examples_per_epoch = self.num_examples_per_epoch

        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return self._generate_image_and_label_batch(
            image, label, min_queue_examples, batch_size, shuffle=False)
