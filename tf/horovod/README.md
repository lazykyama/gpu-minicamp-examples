# TensorFlow native API Examples

Please refer [../README.md](../README.md) to undrestand how you launch runtime container environment.

## Standard data API as data preprocessing and loading

### Single GPU with standard data API

Example command:

```
cd /ws/gpu-minicamp-examples/tf/horovod
python tf2_keras_nonhvd_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple GPU on single node with standard data API

Example command:

```
cd /ws/gpu-minicamp-examples/tf/horovod
mpirun -np 8 --allow-run-as-root \
    python tf2_keras_hvd_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

Note that `--allow-run-as-root` is necessary for only a container environment.
If you will run this example on other environment like supercomputer, etc, then please remove it.

### Multiple nodes with standard data API

Example command for two nodes:

```
mpirun -np 16 --allow-run-as-root -wdir /ws/gpu-minicamp-examples/tf/horovod \
    python tf2_keras_hvd_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

Regarding `-wdir` option details, please see ["Current Working Directory" section](https://www.open-mpi.org/doc/v4.1/man1/mpirun.1.php#sect16) in MPI document.
Also, other options should be added to `mpirun` are strongly depending on your computing cluster.
Please reach out to your system admins.

## DALI as data preprocessing and loading

### Single GPU with DALI

Example command:

```
cd /ws/gpu-minicamp-examples/tf/horovod/dali
python tf2_keras_nonhvd_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple GPU on single node with DALI

Example command:

```
cd /ws/gpu-minicamp-examples/tf/horovod/dali
mpirun -np 8 --allow-run-as-root \
    python tf2_keras_hvd_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```

### Multiple nodes with DALI

Example command for two nodes:

```
mpirun -np 16 --allow-run-as-root -wdir /ws/gpu-minicamp-examples/tf/horovod/dali \
    python tf2_keras_hvd_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models
```
