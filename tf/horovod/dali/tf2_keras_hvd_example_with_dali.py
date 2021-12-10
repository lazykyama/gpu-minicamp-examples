# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This code is baed on official example of Horovod below.
# https://github.com/horovod/horovod/blob/2481cbf7046143ae0981c58c46795768529892c5/examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py
#
# Mainly, several changes are applied to this example:
#    - Dataset is not generated in runtime.
#      It consists of randomly generated "files".
#    - The number of batches for each iter is defined based on
#      the actual dataset size.
#    - Simplified DistributedOptimizer options.
#    - Validation is enabled.
#    - Model saving part is added.
#    - etc.
#
# Original example is distributed under the Apache License, Version 2.0
# like below.
#
# ==============================================================================
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse
import glob
import os
import math
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf
import horovod.tensorflow.keras as hvd
from horovod.tensorflow.mpi_ops import Average as mpi_avg_op

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.plugin.tf as dali_tf


def main():
    # Example settings
    parser = argparse.ArgumentParser(
        description="TensorFlow2 Keras Horovod Example",
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
        "--output-path",
        type=str,
        default="./models",
        help="output path to store saved model",
    )

    parser.add_argument(
        "--no-validation", action="store_true", help="Disable validation."
    )

    args = parser.parse_args()
    device = "GPU"

    # Horovod: initialize Horovod.
    hvd.init()

    if hvd.rank() == 0:
        print("Batch size: %d" % args.batch_size)
        print("Number of %ss: %d" % (device, hvd.size()))

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(
            gpus[hvd.local_rank()], "GPU"
        )

    # Load dataset from randomly generated files.
    def prepare_dataset(
        args, batch_size, subdir, return_n_classes=False, shuffle=True
    ):
        parentdir = os.path.join(args.input_path, subdir)
        if return_n_classes:
            n_classes = len(glob.glob(os.path.join(parentdir, "cls_*")))
        n_data_size = len(glob.glob(os.path.join(parentdir, "cls_*", "*.jpg")))
        n_sharded_size = (
            n_data_size // hvd.size()
        )  # NOTE: simplified calculation.

        # Build DALI data loading pipeline.
        # This pipeline will do: 1) reading file, 2) decoding jpeg,
        # 3) normalizing values, and 4) transferring data from CPU to GPU.
        @pipeline_def
        def _build_pipeline(shard_id, num_shards):
            # NOTE: The `fn.readers.file()` returns label IDs,
            #     : not label name directly extracted from directory path.
            #     : For example, cls_0000000 -> 0.
            img_files, labels = fn.readers.file(
                file_root=parentdir,
                random_shuffle=shuffle,
                name="FilesReader",
                shard_id=shard_id,
                num_shards=num_shards,
            )
            images = fn.decoders.image(img_files, device="mixed")
            images = fn.normalize(images, device="gpu")
            return images, labels.gpu()

        # Make dataset with DALIDataset.
        shapes = ((batch_size, 224, 224, 3), (batch_size,))
        dtypes = (tf.float32, tf.int32)

        device_id = hvd.local_rank()
        with tf.device(f"/gpu:{device_id}"):
            dali_pipeline = _build_pipeline(
                batch_size=batch_size,
                device_id=device_id,
                shard_id=hvd.rank(),
                num_shards=hvd.size(),
            )
            dataset = dali_tf.DALIDataset(
                pipeline=dali_pipeline,
                batch_size=batch_size,
                output_shapes=shapes,
                output_dtypes=dtypes,
                device_id=device_id,
            )

        if return_n_classes:
            return dataset, n_sharded_size, n_classes
        else:
            return dataset, n_sharded_size

    train_ds, n_sharded_train_ds, n_classes = prepare_dataset(
        args, args.batch_size, "train", return_n_classes=True
    )
    num_batches_per_epoch = math.ceil(n_sharded_train_ds / args.batch_size)
    if args.no_validation:
        val_ds = None
        num_val_batches_per_epoch = None
    else:
        val_ds, n_sharded_val_ds = prepare_dataset(
            args, args.batch_size, "val", shuffle=False
        )
        num_val_batches_per_epoch = math.ceil(
            n_sharded_val_ds / args.batch_size
        )

    # Set up standard model.
    def build_model(n_classes):
        base_model = tf.keras.applications.ResNet50(
            weights=None, include_top=False
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        return model

    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    model = build_model(n_classes)
    opt = tf.optimizers.SGD(args.lr)

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt)

    # Horovod: Specify `experimental_run_tf_function=False` to
    # ensure TensorFlow uses hvd.DistributedOptimizer() to compute gradients.
    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        experimental_run_tf_function=False,
    )

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to
        # all other processes.
        # This is necessary to ensure consistent initialization of
        # all workers when training is started with random weights or
        # restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    class TimingCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.img_secs = []

        def on_train_end(self, logs=None):
            img_sec_mean = np.mean(self.img_secs)
            img_sec_conf = 1.96 * np.std(self.img_secs)
            print(
                "Img/sec per %s: %.1f +-%.1f"
                % (device, img_sec_mean, img_sec_conf)
            )
            print(
                "Total img/sec on %d %s(s): %.1f +-%.1f"
                % (
                    hvd.size(),
                    device,
                    hvd.size() * img_sec_mean,
                    hvd.size() * img_sec_conf,
                )
            )

        def on_epoch_begin(self, epoch, logs=None):
            self.starttime = timer()

        def on_epoch_end(self, epoch, logs=None):
            time = timer() - self.starttime
            img_sec = args.batch_size * num_batches_per_epoch / time
            print("Iter #%d: %.1f img/sec per %s" % (epoch, img_sec, device))
            # skip warm up epoch
            if epoch > 0:
                self.img_secs.append(img_sec)

    class AverageLossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs["val_loss"]
            agged_val_loss = hvd.allreduce(val_loss, op=mpi_avg_op)
            if hvd.rank() == 0:
                print(f"aggregated val_loss = {agged_val_loss:0.5f}")

    # Horovod: calculate average validation loss, and write logs on worker 0.
    if not args.no_validation:
        callbacks.append(AverageLossCallback())
    if hvd.rank() == 0:
        timing = TimingCallback()
        callbacks.append(timing)

    # Train the model.
    model.fit(
        train_ds,
        steps_per_epoch=num_batches_per_epoch,
        validation_data=val_ds,
        validation_steps=num_val_batches_per_epoch,
        callbacks=callbacks,
        epochs=args.num_epochs,
        verbose=1 if hvd.rank() == 0 else 0,
    )

    if hvd.rank() == 0:
        # Save model.
        model.save(args.output_path)


if __name__ == "__main__":
    main()
