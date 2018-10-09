#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dicom-dir> <label-file> <output-dir>"
    exit 1
fi
docker run --runtime=nvidia --rm -v `pwd`/"$1":/dicom_dir -v `pwd`/`dirname $2`:/label -v `pwd`/"$3":/records yukw777/tf-models python research/object_detection/dataset_tools/create_pneumonia_tf_record.py --dicom_dir /dicom_dir --label_file /label/`basename $2` --output_dir /records
