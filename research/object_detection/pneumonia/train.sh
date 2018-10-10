#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <data-dir> <config-file> <results-dir> <available-gpus>"
    exit 1
fi
docker run -e "NVIDIA_VISIBLE_DEVICES=$4" --runtime=nvidia --rm -v `pwd`/"$1":/data -v `pwd`/`dirname $2`:/model -v `pwd`/"$3":/results yukw777/tf-models python research/object_detection/model_main.py --pipeline_config_path=/model/`basename $2` --model_dir=/results
