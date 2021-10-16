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

## Performance guideline

A few reference information is listed below.
You can use these information to verify if the code is properly running on your system.

* PyTorch
    - Standard example should achieve ~6x faster performance on 8 GPUs compared to 1 GPU.
    - When using DALI, the performance gain ratio should be increased to x7.2.
    - The performance can be more improved in the case of `batchsize=128` instead of `batchsize=64`.
* TensorFlow
    - Standard example should achieve ~3.6x faster performance on 8 GPUs (in the case of `MirroredStrategy`) compared to 1 GPU.
    - When using DALI, the performance gain ratio should be increased to x5.8.
* TensorFlow+Horovod
    - Standard example should achieve ~4.6x faster performance on 8 GPUs compared to 1 GPU.
    - When using DALI, the performance gain ratio should be increased to x6.9.
    - The performance can be slightly improved in the case of `batchsize=128` instead of `batchsize=64`.
