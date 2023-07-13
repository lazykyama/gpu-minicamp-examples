# GPU-minicamp-examples

このリポジトリは、各スーパーコンピュータセンターと共催される、GPUミニキャンプで利用されるサンプルコードのリポジトリです。
マルチGPUおよび、マルチノードで実行されるディープラーニングの学習用スクリプトの書き方について理解を深めることを主な目的として、設計されています。

## 動作確認済みの環境

* PyTorchのサンプルは `nvcr.io/nvidia/pytorch:23.06-py3` のコンテナでテストされています。
* TensorFlowのサンプルは `nvcr.io/nvidia/tensorflow:21.05-tf2-py3` のコンテナでテストされています。
    - TensorFlowのネイティブAPIのサンプルは、新しめのコンテナで実行することもできますが、最後のログが表示されたのち、`terminate called without an active exception` というエラーが表示されるためご注意ください。(性能などへの影響はないものと思われます)
    - TensorFlow 2.4.0およびそれ以前 (NGCコンテナとしては21.05かそれ以前) であれば問題なさそうです。

## データの準備

このリポジトリのサンプルは、`make_pseudo_data.py` を使って生成された議事データを利用する前提となっています。
以下のコマンドは実行例です。

```
python make_pseudo_data.py --num-images 100000 --num-classes 100 --outdir /path/to/your/datadir/ --val-ratio 0.2
```

NGCのTensorFlowコンテナで実行する場合、`pillow` をインストールする必要があります。

## PyTorch examples

詳細は [pytorch/README_ja.md](pytorch/README_ja.md) を参照ください。

## TensorFlow examples

詳細は [tf/README_ja.md](tf/README_ja.md) を参照ください。

## TensorFlow+Horovod examples

詳細は [tf/README_ja.md](tf/README_ja.md) を参照ください。

## パフォーマンスガイドライン

以下にいくつかの参考情報を記載しています。
利用されるシステム上でコードが適切に動作していそうか確認する際など、ご活用ください。

* PyTorch
    - 基本のサンプルコードは、1GPUでの実行に対して、8GPUの場合、6倍程度高速になるはずです。
    - DALIを利用すると、7.2倍まで高速化率が上昇するはずです。
    - バッチサイズは、`batchsize=64` より `batchsize=128` のほうが、さらに性能が上がります。
* TensorFlow
    - 基本のサンプルコードは、1GPUでの実行に対して、8GPU (`MirroredStrategy` 利用) の場合、3.6倍程度の高速化になるはずです。
    - DALIを利用することで、5.8倍まで上昇するはずです。
* TensorFlow+Horovod
    - 基本のサンプルコードは、1GPUでの実行に対して、8GPUの場合、4.6倍程度高速になるはずです。
    - DALIにより、6.9倍まで上昇するはずです。
    - バッチサイズは、`batchsize=64` より `batchsize=128` としたほうが、若干性能が上がります。
