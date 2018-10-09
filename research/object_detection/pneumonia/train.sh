#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <data-dir> <config-file> <results-dir>"
    exit 1
fi
docker run --runtime=nvidia --rm -v `pwd`/"$1":/data -v `pwd`/`dirname $2`:/model -v `pwd`/"$3":/results yukw777/tf-models python research/object_detection/model_main.py --pipeline_config_path=/model/`basename $2` --model_dir=/results
