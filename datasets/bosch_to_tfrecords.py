# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
https://hci.iwr.uni-heidelberg.de/node/6132
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf

# import xml.etree.ElementTree as ET
import yaml

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
# from datasets.pascalvoc_common import VOC_LABELS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'train.yaml'
#DIRECTORY_IMAGES = 'JPEGImages/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200


def _process_image(dataset_dir, img_info):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """

# {
#     'path': './rgb/train/2015-10-05-16-02-30_bag/720308.png',
#     'boxes': [
#         {
#             'y_min': 1.3319230377,
#             'occluded': False,
#             'x_max': 544.6893516913,
#             'y_max': 71.6809366948,
#             'label': 'Green',
#             'x_min': 508.9382136033
#         }, {
#             'y_min': 154.3928767198,
#             'occluded': False,
#             'x_max': 892.9434568191,
#             'y_max': 224.5744078958,
#             'label': 'Green',
#             'x_min': 861.1518230386
#         }
#     ]
# }

    # Read the image file.
    # filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    filename = os.path.join(dataset_dir, img_info['path'])
    image_data = tf.gfile.FastGFile(filename+'.jpg', 'rb').read()

    # Read the XML annotation file.
    # filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    # tree = ET.parse(filename)
    # root = tree.getroot()

    # Image shape.
    # size = root.find('size')
    shape = [720,
             1280,
             3]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    # for obj in root.findall('object'):
    for box_info in img_info['boxes']:
        if (
            box_info['y_min'] >= 0 and
            box_info['x_min'] >= 0 and
            box_info['y_max'] < shape[0] and
            box_info['x_max'] < shape[1]
        ):

            # label = obj.find('name').text
            # labels.append(int(VOC_LABELS[label][0]))
            # labels_text.append(label.encode('ascii'))

            labels.append(1)
            labels_text.append('traffic light'.encode('ascii'))

            # if obj.find('difficult'):
            #     difficult.append(int(obj.find('difficult').text))
            # else:
            difficult.append(0)
            # if obj.find('truncated'):
            #     truncated.append(int(obj.find('truncated').text))
            # else:
            truncated.append(0)

            # bbox = obj.find('bndbox')
            bboxes.append((float(box_info['y_min']) / shape[0],
                           float(box_info['x_min']) / shape[1],
                           float(box_info['y_max']) / shape[0],
                           float(box_info['x_max']) / shape[1]
                           ))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, img_info, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, img_info)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())

    return len(bboxes)

def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='bosch_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    annotations_yaml_path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    with open(annotations_yaml_path, 'r') as f:
        annotations_yaml = f.read()

    annotations = yaml.load(annotations_yaml)
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(annotations)

    # Process dataset files.
    i = 0
    fidx = 0
    bboxes_count = 0
    while i < len(annotations):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(annotations) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(annotations)))
                sys.stdout.flush()

                img_info = annotations[i]
                if len(img_info['boxes']) > 0:
                    bboxes_count += _add_to_tfrecord(dataset_dir, img_info, tfrecord_writer)

                # filename = filenames[i]
                # img_name = filename[:-4]
                i += 1
                j += 1

            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the bosch small traffic lights dataset!')
    print('\n bboxes count: {}'.format(bboxes_count))
