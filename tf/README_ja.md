# TensorFlow Examples

TensorFlowのサンプルは以下の2パートから構成されています:

* ネイティブAPI
    - [native/README_ja.md](native/README_ja.md).
* Horovod
    - [horovod/README_ja.md](horovod/README_ja.md).

いずれも、以下のように起動されたコンテナ上で実行されることを想定しています。
Horovodのサンプルでは、データロードまわりのエラーを回避するため `--privileged` オプションが必要となります。

```
cd /path/to/your/workdir/
docker run --gpus=all --rm -ti --privileged --ipc=host -v $(pwd):/ws nvcr.io/nvidia/tensorflow:21.05-tf2-py3
```

また、ディレクトリ構造は以下のものを想定しています。

```
/ws
├── data
└── gpu-minicamp-examples
```
