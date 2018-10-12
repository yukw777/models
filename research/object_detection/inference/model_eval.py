# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Infers detections on a TFRecord of TFExamples given an inference graph.

Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_images_dir=/path/to/output/detections/images \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb

The output is a collection of jpeg images. Each image has the ground truth boxes
drawn, as well as the inferred bounding boxes.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.
"""

import itertools
import tensorflow as tf
import numpy as np
from object_detection.inference import detection_inference
from object_detection.core import standard_fields
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
from io import BytesIO

tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
FLAGS = tf.flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                    'inference_graph']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  with tf.Session() as sess:
    input_tfrecord_paths = [
        v for v in FLAGS.input_tfrecord_paths.split(',') if v]
    tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
    serialized_example_tensor, image_tensor = detection_inference.build_input(
        input_tfrecord_paths)
    tf.logging.info('Reading graph and building model...')
    (detected_boxes_tensor, detected_scores_tensor,
     detected_labels_tensor) = detection_inference.build_inference_graph(
         image_tensor, FLAGS.inference_graph)

    tf.logging.info('Running inference and writing output to {}'.format(
        FLAGS.output_tfrecord_path))
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
    try:
      for counter in itertools.count():
        tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10, counter)
        tf_example = tf.train.Example()
        (serialized_example, detected_boxes, detected_scores,
          detected_classes) = sess.run([
              serialized_example_tensor, detected_boxes_tensor, detected_scores_tensor,
              detected_labels_tensor
          ])
        tf_example.ParseFromString(serialized_example)
        encoded_jpg = tf_example.features.feature[standard_fields.TfExampleFields.image_encoded].bytes_list.value[0]
        image = Image.open(BytesIO(encoded_jpg))
        image_np = np.array(image)
        image_np = np.stack([image_np] * 3, axis=2)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            detected_boxes,
            detected_classes,
            detected_scores,
            {1: {'id': 1, 'name': 'pneumonia'}},
            use_normalized_coordinates=True,
            max_boxes_to_draw=3,
            min_score_thresh=0.0001
        )
        im = Image.fromarray(image_np)
        im.save("test.jpg")
    except tf.errors.OutOfRangeError:
      tf.logging.info('Finished processing records')


if __name__ == '__main__':
  tf.app.run()
