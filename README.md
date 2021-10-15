# GPU-minicamp-examples

This repository contains several example codes which can be used in GPU minicamp on supercomputer. Main purpose of this repo is to explain how you can implement multi-node deep learning training scripts.

## Environment

* PyTorch example is tested on `nvcr.io/nvidia/pytorch:21.09-py3`.
* TensorFlow example is tested on `nvcr.io/nvidia/tensorflow:21.05-tf2-py3`.
    - Note that you can run TensorFlow's native API example on newer version container images. But, after last log message of each script, you may see a strange error message, `terminate called without an active exception`.
    - TensorFlow 2.4.0 or earlier (=NGC 21.05 or earlier) looks like ok.

## Data preparation

This example assumes that the users make pseudo dataset in advance with a script, `make_pseudo_data.py`.
Following command is an usage example.

```
python make_pseudo_data.py --num-images 100000 --num-classes 100 --outdir /path/to/your/datadir/ --val-ratio 0.2
```

You need to install `pillow` when you run this script in NGC TensorFlow container.

## PyTorch examples

Details are described at [pytorch/README.md](pytorch/README.md).

## TensorFlow examples

Details are described at [tf/README.md](tf/README.md).

## TensorFlow+Horovod examples

Details are described at [tf/README.md](tf/README.md).
