# TensorFlow Examples

[Japanese version readme](./README_ja.md)

TensorFlow examples consist of two parts:

* native API
    - [native/README.md](native/README.md).
* Horovod
    - [horovod/README.md](horovod/README.md).

Both examples assume that your codes will be running on a container launched by commends below.
Note that an option, `--privileged`, is required to avoid several errors on data reading in Horovod examples.

```
cd /path/to/your/workdir/
docker run --gpus=all --rm -ti --privileged --ipc=host -v $(pwd):/ws nvcr.io/nvidia/tensorflow:21.05-tf2-py3
```

Also, it assumes following directory structure.

```
/ws
├── data
└── gpu-minicamp-examples
```
