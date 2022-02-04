# TensorFlow+Horovod Examples

ランタイムコンテナの起動方法については、[../README_ja.md](../README_ja.md) を参照してください。

## 標準的なデータAPIによる、データロードおよび前処理

### シングルGPU

スクリプト: [tf2_keras_nonhvd_example.py](tf2_keras_nonhvd_example.py)

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/tf/horovod
python tf2_keras_nonhvd_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### シングルノードマルチGPU

スクリプト: [tf2_keras_hvd_example.py](tf2_keras_hvd_example.py)

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/tf/horovod
mpirun -np 8 --allow-run-as-root \
    python tf2_keras_hvd_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

コンテナでの実行には `--allow-run-as-root` が必要となります。
その他の環境で実行する場合、不要なことが多いです。

### マルチノード

スクリプト: [tf2_keras_hvd_example.py](tf2_keras_hvd_example.py)

2ノードでのサンプルコマンド:

```
mpirun -np 16 --allow-run-as-root -wdir /ws/gpu-minicamp-examples/tf/horovod \
    python tf2_keras_hvd_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

`-wdir` オプションの詳細については、MPIのドキュメントにある ["Current Working Directory" section](https://www.open-mpi.org/doc/v4.1/man1/mpirun.1.php#sect16) を参照してください。
また、`mpirun` にどんなオプションが指定されるべきか、という点については、各計算機の環境に強く依存します。
システム管理者などにご相談ください。

## DALIによる、データロードおよび前処理

### シングルGPU

スクリプト: [tf2_keras_nonhvd_example_with_dali.py](dali/tf2_keras_nonhvd_example_with_dali.py)

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/tf/horovod/dali
python tf2_keras_nonhvd_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### シングルノードマルチGPU

スクリプト: [tf2_keras_hvd_example_with_dali.py](dali/tf2_keras_hvd_example_with_dali.py)

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/tf/horovod/dali
mpirun -np 8 --allow-run-as-root \
    python tf2_keras_hvd_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### マルチノード

スクリプト: [tf2_keras_hvd_example_with_dali.py](dali/tf2_keras_hvd_example_with_dali.py)

2ノードでのサンプルコマンド:

```
mpirun -np 16 --allow-run-as-root -wdir /ws/gpu-minicamp-examples/tf/horovod/dali \
    python tf2_keras_hvd_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```
