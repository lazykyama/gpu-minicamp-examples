# TensorFlow native API Examples

Please refer [../README.md](../README.md) to undrestand how you launch runtime container environment.

## Standard data API as data preprocessing and loading

### Single GPU

Example command:

```
cd /ws/gpu-minicamp-examples/tf/native
python tf2_keras_singlegpu.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple GPU on single node

Example command:

```
cd /ws/gpu-minicamp-examples/tf/native
python tf2_keras_mirrored_strategy_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple nodes

...

## DALI as data preprocessing and loading

### Single GPU

Example command:

```
cd /ws/gpu-minicamp-examples/tf/native/dali
python tf2_keras_singlegpu_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple GPU on single node

Example command:

```
cd /ws/gpu-minicamp-examples/tf/native/dali
python tf2_keras_mirrored_strategy_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple nodes

This example doesn't exist due to a difficulty about a combination of `MultiWorkerMirroredStrategy` and DALI with `tf.keras` high level API.
If your code is based on custom training, it's not a problem.
