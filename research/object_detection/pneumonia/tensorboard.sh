#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <logdir>"
    exit 1
fi
docker run -d --runtime=nvidia --rm -p 6006:6006 -v `pwd`/"$1":/logdir yukw777/tf-models tensorboard --logdir /logdir
