# PyTorch Examples

Following example instructions assume that your codes will be running on a container launched by commends below.

```
cd /path/to/your/workdir/
docker run --gpus=all --rm -it --ipc=host -v $(pwd):/ws nvcr.io/nvidia/pytorch:21.09-py3
```

Also, it assumes following directory structure.

```
/ws
├── data
└── gpu-minicamp-examples
```

## Standard data API as data preprocessing and loading

### Single GPU

A script: [pytorch_singlegpu_run_example.py](native/pytorch_singlegpu_run_example.py).

Example command:

```
cd /ws/gpu-minicamp-examples/pytorch/native
python pytorch_singlegpu_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 250
```

### Multiple GPU on single node

For multi-GPU training, this code utilizes new version API, `torch.distibuted.run`, introduced since PyTorch 1.9.
If you need to use older API, `torch.distributed.launch`, please add `--use-older-api` option.
(In addition, it is necessary to change `--standalone` or `--rdzv_endpoint` to `--master_addr` option.)
Internally, a slightly different code path is enabled.

A script: [pytorch_distributed_run_example.py](native/pytorch_distributed_run_example.py).

Example command:

```
cd /ws/gpu-minicamp-examples/pytorch/native
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --standalone \
    pytorch_distributed_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

If you are interested in [Torch Distributed Elastic](https://pytorch.org/docs/stable/distributed.elastic.html), also please try a following command.

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --rdzv_id="TestJob" --rdzv_backend=c10d --rdzv_endpoint=localhost \
    pytorch_distributed_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

### Multiple nodes

A script: [pytorch_distributed_run_example.py](native/pytorch_distributed_run_example.py).

Example command:

```
cd /ws/gpu-minicamp-examples/pytorch/native
python -m torch.distributed.run --nnodes=${NUM_YOUR_NODES} --nproc_per_node=8 \
    --rdzv_id="TestJob" --rdzv_backend=c10d --rdzv_endpoint= \
        pytorch_distributed_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

## DALI as data preprocessing and loading

### Single GPU

A script: [pytorch_singlegpu_run_example_with_dali.py](native/dali/pytorch_singlegpu_run_example_with_dali.py).

Example command:

```
cd /ws/gpu-minicamp-examples/pytorch/native/dali
python pytorch_singlegpu_run_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 250
```

### Multiple GPU

A script: [pytorch_distributed_run_example_with_dali.py](native/dali/pytorch_distributed_run_example_with_dali.py).

Example command:

```
cd /ws/gpu-minicamp-examples/pytorch/native/dali
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --standalone \
    pytorch_distributed_run_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

### Multiple nodes

Example command:

```
TBD
```
