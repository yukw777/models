import argparse
import tensorflow as tf

from utils import label_map_util
from builders.dataset_builder import build
from object_detection.protos import input_reader_pb2
from tqdm import tqdm

from model_infer import (
    load_tf_graph,
)


def read_tf_records(eval_file, label_map_path, batch_size):
  input_reader_config = input_reader_pb2.InputReader()
  input_reader_config.label_map_path = label_map_path
  input_reader_config.shuffle = False
  input_reader_config.num_epochs = 1
  input_reader_config.tf_record_input_reader.input_path.extend([eval_file])

  dataset = build(input_reader_config, batch_size=batch_size)
  iterator = dataset.make_initializable_iterator()
  next_batch = iterator.get_next()
  with tf.Session() as sess:
    sess.run(iterator.initializer)
    try:
      while True:
        yield sess.run(next_batch)
    except tf.errors.OutOfRangeError:
      return


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('frozen_graph', help='path to the exported frozen graph')
  parser.add_argument('label_map', help='path to the label map')
  parser.add_argument('label_file', help='path to file with ground truth labels in the Kaggle format')
  parser.add_argument('eval_file', help='glob for the tensorflow evaluation files')
  parser.add_argument('output_dir', help='path to dir for evaluated images')
  parser.add_argument('-b', '--batch-size', help='test batch size, default 8', type=int, default=8)
  parser.add_argument('--max-boxes', help='max boxes to print, default 10', type=int, default=10)
  args = parser.parse_args()

  # load the categories
  category_index = label_map_util.create_category_index_from_labelmap(args.label_map, use_display_name=True)

  # load the frozen graph
  frozen_graph = load_tf_graph(args.frozen_graph)

  for batch in read_tf_records(args.eval_file, args.label_map, args.batch_size):
    import pdb; pdb.set_trace()
