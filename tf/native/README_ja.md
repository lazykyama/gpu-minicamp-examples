# TensorFlow native API Examples

ランタイムコンテナの起動方法については、[../README_ja.md](../README_ja.md) を参照してください。

## 標準的なデータAPIによる、データロードおよび前処理

### シングルGPU

スクリプト: [tf2_keras_singlegpu.py](tf2_keras_singlegpu.py).

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/tf/native
python tf2_keras_singlegpu.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### シングルノードマルチGPU

スクリプト: [tf2_keras_mirrored_strategy_example.py](tf2_keras_mirrored_strategy_example.py).

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/tf/native
python tf2_keras_mirrored_strategy_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### マルチノード

スクリプト: [tf2_keras_multiworker_example.py](tf2_keras_multiworker_example.py).

2ノードでのサンプルコマンド:

ノード0:

```
python tf2_keras_multiworker_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models \
    --worker-addrs ${NODE0_IP_OR_HOSTNAME} ${NODE1_IP_OR_HOSTNAME} --worker-id 0
```

ノード1:

```
python tf2_keras_multiworker_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models \
    --worker-addrs ${NODE0_IP_OR_HOSTNAME} ${NODE1_IP_OR_HOSTNAME} --worker-id 1
```

## DALIによる、データロードおよび前処理

### シングルGPU

スクリプト: [tf2_keras_singlegpu.py](dali/tf2_keras_singlegpu_with_dali.py).

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/tf/native/dali
python tf2_keras_singlegpu_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### シングルノードマルチGPU

スクリプト: [tf2_keras_mirrored_strategy_example.py](dali/tf2_keras_mirrored_strategy_example_with_dali.py).

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/tf/native/dali
python tf2_keras_mirrored_strategy_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### マルチノード

`tf.keras` のハイレベルAPIと、`MultiWorkerMirroredStrategy` およびDALIの組み合わせに課題があるため、DALI利用時のマルチノードサンプルはありません。
カスタムトレーニングを利用した学習の実装を行っている場合、特に問題なく利用できます。
