# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import glob
import os
import time

import torch

import torchvision.models as models

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.types import DALIDataType


def _check_pytorch_version():
    version_info = tuple(map(int, torch.__version__.split(".")[:2]))
    if version_info[0] not in (1, 2):
        # Not supported version.
        return False
    return True


def main():
    is_expected_version = _check_pytorch_version()
    if not is_expected_version:
        raise RuntimeError(
            (
                "Your PyTorch version is not expected in this example code."
                "1.x or 2.x are required."
            )
        )

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare output directory.
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.isdir(args.output_path):
        raise RuntimeError(
            f"{args.output_path} exists, but it is not a directory."
        )

    trainiter, valiter, n_classes = prepare_dataset(
        args.input_path,
        args.batch_size,
        num_workers=args.num_workers,
        no_validation=args.no_validation,
    )

    model = build_model(n_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # NOTE:
    # If you are interested in the acceleration by Tensor Cores,
    # please read the following doc.
    # https://pytorch.org/docs/stable/amp.html
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")

        running_loss = 0.0
        # NOTE:
        # This is a simplified way to measure the time.
        # You should use more precise method to know the performance.
        starttime = time.perf_counter()
        for i, data in enumerate(trainiter):
            data = data[0]
            inputs = data["data"]
            # NOTE:
            # It looks like DALIIterator returns labels with shape=(N, 1).
            # But, PyTorch assumes (N,) is a shape as a label tensor.
            # Unnecessary dimension will be removed by squeeze().
            # NOTE:
            # DALI also has similar function, fn.squeeze(),
            # but it didn't work well at least in the pipeline.
            labels = data["label"].squeeze()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate and show current average loss.
            running_loss += loss.item()
            if (i % args.logging_interval) == (args.logging_interval - 1):
                print(
                    (
                        f"\t [iter={i+1:05d}] training loss = "
                        f"{running_loss / (i+1):.3f}"
                    )
                )
        # End of each epoch.

        # Calculate validation result.
        if not args.no_validation:
            running_valloss = 0.0
            n_valiter = 0
            model.eval()
            with torch.no_grad():
                for valdata in valiter:
                    valdata = valdata[0]
                    val_in = valdata["data"]
                    val_label = valdata["label"].squeeze()
                    valout = model(val_in)
                    valloss = criterion(valout, val_label)
                    running_valloss += valloss.item()
                    n_valiter += len(val_in)
            model.train()

        # Show this epoch time and training&validation losses.
        # NOTE: This time includes vaidation time.
        duration = time.perf_counter() - starttime
        if args.no_validation:
            print(
                (
                    f"\t [iter={i+1:05d}] "
                    f"{duration:.3f}s {duration*1000. / i:.3f}ms/step, "
                    f"loss = {running_loss / (i+1):.3f}"
                )
            )
        else:
            print(
                (
                    f"\t [iter={i+1:05d}] "
                    f"{duration:.3f}s {duration*1000. / i:.3f}ms/step, "
                    f"loss = {running_loss / (i+1):.3f}, "
                    f"val_loss = {running_valloss / n_valiter:.3f}"
                )
            )

    # Save model.
    model_filepath = os.path.join(args.output_path, "model.pth")
    torch.save(model.state_dict(), model_filepath)

    print("done.")


def prepare_dataset(datadir, batch_size, num_workers=4, no_validation=False):
    parentdir = os.path.join(datadir, "train")
    n_classes = len(glob.glob(os.path.join(parentdir, "cls_*")))
    n_data = len(glob.glob(os.path.join(parentdir, "cls_*", "*.jpg")))

    # Build DALI data loading pipeline.
    # This pipeline will do: 1) reading file, 2) decoding jpeg,
    # 3) normalizing values, and 4) transferring data from CPU to GPU.
    @pipeline_def
    def _build_pipeline(rootdir, shuffle):
        # NOTE:
        # The `fn.readers.file()` returns label IDs,
        # not label name directly extracted from directory path.
        # For example, cls_0000000 -> 0.
        img_files, labels = fn.readers.file(
            file_root=rootdir, random_shuffle=shuffle
        )
        images = fn.decoders.image(img_files, device="mixed")
        images = fn.normalize(images, device="gpu")
        # NOTE:
        # fn.decoders.image returns NHWC layout tensor.
        # PyTorch assumes NCHW layout.
        # fn.transpose will convert this layout.
        images = fn.transpose(images, perm=[2, 0, 1], device="gpu")
        # NOTE:
        # The dtype of labels returned by DALI is int32 in default.
        # But, int32 is usually unsupported by many PyTorch opperations.
        # It's also necessary to convert data type into int64.
        labels = fn.cast(labels, dtype=DALIDataType.INT64)
        return images, labels.gpu()

    train_dali_pipeline = _build_pipeline(
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0,
        rootdir=parentdir,
        shuffle=True,
    )
    train_iterator = DALIClassificationIterator(
        train_dali_pipeline,
        size=n_data,
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        last_batch_padded=False,
    )

    # Prepare validation data iterator.
    if no_validation:
        val_iterator = None
    else:
        val_file_pattern = os.path.join(datadir, "val", "cls_*", "*.jpg")
        n_data = len(glob.glob(val_file_pattern))
        val_dali_pipeline = _build_pipeline(
            batch_size=batch_size,
            num_threads=num_workers,
            device_id=0,
            rootdir=parentdir,
            shuffle=False,
        )
        val_iterator = DALIClassificationIterator(
            val_dali_pipeline,
            size=n_data,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            last_batch_padded=False,
        )

    return train_iterator, val_iterator, n_classes


def build_model(n_classes):
    model = models.resnet50(pretrained=False)
    n_fc_in_feats = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(n_fc_in_feats, 512), torch.nn.Linear(512, n_classes)
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch torch.distributed.run Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="./images",
        help="a parent directory path to input image files",
    )

    parser.add_argument(
        "--batch-size", type=int, default=64, help="input batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10, help="number of epochs"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="number of workers for data loading",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./models",
        help="output path to store saved model",
    )

    parser.add_argument(
        "--no-validation", action="store_true", help="Disable validation."
    )

    parser.add_argument(
        "--logging-interval", type=int, default=10, help="logging interval"
    )

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main()
