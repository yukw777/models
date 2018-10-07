import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import os
import pydicom

from utils import label_map_util
from object_detection.utils import ops as utils_ops


def load_dcm_into_numpy_array(dcm):
  im = np.stack([dcm.pixel_array] * 3, axis=2)
  im_width, im_height = dcm.pixel_array.shape
  return im.reshape((im_height, im_width, 3))


def run_inference_for_image_batch(batch, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, batch[0].shape[0], batch[0].shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: batch})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = [int(d) for d in output_dict['num_detections']]
      output_dict['detection_classes'] = [c.astype(np.uint8) for c in output_dict['detection_classes']]
  return output_dict


def run_inference_for_single_image(image, graph):
  batch_output_dict = run_inference_for_image_batch(np.expand_dims(image, 0), graph)
  output_dict = {}
  output_dict['num_detections'] = batch_output_dict['num_detections'][0]
  output_dict['detection_classes'] = batch_output_dict['detection_classes'][0]
  output_dict['detection_boxes'] = batch_output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = batch_output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = batch_output_dict['detection_masks'][0]
  return output_dict

def load_tf_graph(graph):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(graph, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph


def batchify(l, batch_size):
  return [l[i:i + batch_size] for i in range(0, len(l), batch_size)]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('frozen_graph', help='path to the exported frozen graph')
  parser.add_argument('label_map', help='path to the label map')
  parser.add_argument('test_dir', help='path to directory with test DICOMs')
  parser.add_argument('submission', help='path to the Kaggle submission file')
  args = parser.parse_args()

  # load the categories
  category_index = label_map_util.create_category_index_from_labelmap(args.label_map, use_display_name=True)

  # load the frozen graph
  frozen_graph = load_tf_graph(args.frozen_graph)

  # gather test DICOM images
  test_images = glob.glob(os.path.join(args.test_dir, '*.dcm'))

  # run the inference and generate a submission file
  submit_dict = {'patientId': [], 'PredictionString': []}
  for image in test_images:
    dcm = pydicom.read_file(image)
    image_np = load_dcm_into_numpy_array(dcm)
    output = run_inference_for_single_image(image_np, frozen_graph)

    submit_dict['patientId'].append(os.path.splitext(os.path.basename(image))[0])
    boxes = []
    for score, box in zip(output['detection_scores'], output['detection_boxes']):
      im_width, im_height = image_np.shape
      ymin, xmin, ymax, xmax = box
      x = xmin * im_width
      y = ymin * im_height
      w = xmax * im_width - x
      h = ymax * im_height - y
      boxes.append('{0} {1} {2} {3} {4}'.format(score, x, y, w, h))
    submit_dict['PredictionString'].append(' '.join(boxes))

  pd.DataFrame(submit_dict).to_csv(args.submission, index=False)
