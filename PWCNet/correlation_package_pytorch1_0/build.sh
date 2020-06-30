#!/usr/bin/env bash

echo "Need pytorch>=1.0.0"
source activate pytorch1.0.0

export PYTHONPATH=$PYTHONPATH:$(pwd)/../../my_package

rm -rf build *.egg-info dist
/content/Python/bin/python3.6 setup.py install
