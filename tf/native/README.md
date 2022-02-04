# TensorFlow native API Examples

[Japanese version readme](./README_ja.md)

Please refer [../README.md](../README.md) to undrestand how you launch runtime container environment.

## Standard data API as data preprocessing and loading

### Single GPU

A script: [tf2_keras_singlegpu.py](tf2_keras_singlegpu.py).

Example command:

```
cd /ws/gpu-minicamp-examples/tf/native
python tf2_keras_singlegpu.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple GPU on single node

A script: [tf2_keras_mirrored_strategy_example.py](tf2_keras_mirrored_strategy_example.py).

Example command:

```
cd /ws/gpu-minicamp-examples/tf/native
python tf2_keras_mirrored_strategy_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple nodes

A script: [tf2_keras_multiworker_example](tf2_keras_multiworker_example).

Example command for two nodes:

On node 0,

```
python tf2_keras_multiworker_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models \
    --worker-addrs ${NODE0_IP_OR_HOSTNAME} ${NODE1_IP_OR_HOSTNAME} --worker-id 0
```

On node 1,

```
python tf2_keras_multiworker_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models \
    --worker-addrs ${NODE0_IP_OR_HOSTNAME} ${NODE1_IP_OR_HOSTNAME} --worker-id 1
```

## DALI as data preprocessing and loading

### Single GPU

A script: [tf2_keras_singlegpu.py](dali/tf2_keras_singlegpu_with_dali.py).

Example command:

```
cd /ws/gpu-minicamp-examples/tf/native/dali
python tf2_keras_singlegpu_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple GPU on single node

A script: [tf2_keras_mirrored_strategy_example.py](dali/tf2_keras_mirrored_strategy_example_with_dali.py).

Example command:

```
cd /ws/gpu-minicamp-examples/tf/native/dali
python tf2_keras_mirrored_strategy_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple nodes

This example doesn't exist due to a difficulty about a combination of `MultiWorkerMirroredStrategy` and DALI with `tf.keras` high level API.
If your code is based on custom training, it's not a problem.
