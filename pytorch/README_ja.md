# PyTorch Examples

以降のコマンドは、以下のように起動されたコンテナ上で実行されることを想定しています。

```
cd /path/to/your/workdir/
docker run --gpus=all --rm -it --ipc=host -v $(pwd):/ws nvcr.io/nvidia/pytorch:21.09-py3
```

また、ディレクトリ構造は以下のものを想定しています。

```
/ws
├── data
└── gpu-minicamp-examples
```

## 標準的なデータAPIによる、データロードおよび前処理

### シングルGPU

スクリプト: [pytorch_singlegpu_run_example.py](native/pytorch_singlegpu_run_example.py).

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/pytorch/native
python pytorch_singlegpu_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 250
```

### シングルノードマルチGPU

マルチGPU実行のため、PyTorch 1.9から導入された新しいAPIである `torch.distibuted.run` を利用しています。
旧APIである `torch.distributed.launch` を試す必要がある場合、スクリプト実行時に `--use-older-api` オプションを追加してください。
(加えて、`--standalone` もしくは `--rdzv_endpoint` を `--master_addr` に変更する必要もあります。)
内部的には若干異なるコードが利用されます。

スクリプト: [pytorch_distributed_run_example.py](native/pytorch_distributed_run_example.py).

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/pytorch/native
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --standalone \
    pytorch_distributed_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

[Torch Distributed Elastic](https://pytorch.org/docs/stable/distributed.elastic.html) の機能に興味がある場合、以下のコマンドを試してみてください。

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --rdzv_id="TestJob" --rdzv_backend=c10d --rdzv_endpoint=localhost \
    pytorch_distributed_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

### マルチノード

スクリプト: [pytorch_distributed_run_example.py](native/pytorch_distributed_run_example.py).

2ノードでのサンプルコマンド:

ノード0:

```
cd /ws/gpu-minicamp-examples/pytorch/native
python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 \
    --rdzv_id="TestJob" --rdzv_backend=c10d --rdzv_endpoint=${NODE0_IP_OR_HOSTNAME} \
        pytorch_distributed_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

ノード1:

```
cd /ws/gpu-minicamp-examples/pytorch/native
python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 \
    --rdzv_id="TestJob" --rdzv_backend=c10d --rdzv_endpoint=${NODE0_IP_OR_HOSTNAME} \
        pytorch_distributed_run_example.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

## DALIによる、データロードおよび前処理

### シングルGPU

スクリプト: [pytorch_singlegpu_run_example_with_dali.py](native/dali/pytorch_singlegpu_run_example_with_dali.py).

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/pytorch/native/dali
python pytorch_singlegpu_run_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 250
```

### マルチGPU

スクリプト: [pytorch_distributed_run_example_with_dali.py](native/dali/pytorch_distributed_run_example_with_dali.py).

サンプルコマンド:

```
cd /ws/gpu-minicamp-examples/pytorch/native/dali
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --standalone \
    pytorch_distributed_run_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

### マルチノード

スクリプト: [pytorch_distributed_run_example_with_dali.py](native/dali/pytorch_distributed_run_example_with_dali.py).

2ノードでのサンプルコマンド:

ノード0:

```
cd /ws/gpu-minicamp-examples/pytorch/native/dali
python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 \
    --rdzv_id="TestJob" --rdzv_backend=c10d --rdzv_endpoint=${NODE0_IP_OR_HOSTNAME} \
        pytorch_distributed_run_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```

ノード1:

```
cd /ws/gpu-minicamp-examples/pytorch/native/dali
python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 \
    --rdzv_id="TestJob" --rdzv_backend=c10d --rdzv_endpoint=${NODE0_IP_OR_HOSTNAME} \
        pytorch_distributed_run_example_with_dali.py --input-path /ws/data/ --num-epochs 4 --output-path /path/to/models --logging-interval 30
```
