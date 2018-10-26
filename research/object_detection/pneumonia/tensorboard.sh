#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <logdir> <background>"
    exit 1
fi
docker run $2 --runtime=nvidia --rm -p 6006:6006 -v `pwd`/"$1":/logdir yukw777/tf-models tensorboard --logdir /logdir
