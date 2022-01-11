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

# This code is based on an official example of Horovod below.
# https://github.com/horovod/horovod/blob/2481cbf7046143ae0981c58c46795768529892c5/examples/tensorflow2/tensorflow2_keras_synthetic_benchmark.py
# The original example is distributed under the Apache License,
# Version 2.0 like below.
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
#
# This example includes several changes like below:
#    - All Horovod related parts are removed to emphasize differences
#      between Horovod vs non-Horovod codes.
#    - Dataset is not generated in runtime.
#      It consists of randomly generated "files".
#    - DALI is used as a data loading mechanism.
#    - The number of batches for each iter is defined based on
#      the actual dataset size.
#    - Simplified DistributedOptimizer options.
#    - Validation is enabled.
#    - Model saving part is added.
#    - etc.
#


import argparse
import glob
import os
import math
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf

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

    print("Batch size: %d" % args.batch_size)

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")

    # Load dataset from randomly generated files.
    def prepare_dataset(
        args, batch_size, subdir, return_n_classes=False, shuffle=True
    ):
        parentdir = os.path.join(args.input_path, subdir)
        if return_n_classes:
            n_classes = len(glob.glob(os.path.join(parentdir, "cls_*")))
        n_data_size = len(glob.glob(os.path.join(parentdir, "cls_*", "*.jpg")))

        # Build DALI data loading pipeline.
        # This pipeline will do: 1) reading file, 2) decoding jpeg,
        # 3) normalizing values, and 4) transferring data from CPU to GPU.
        @pipeline_def
        def _build_pipeline():
            # NOTE: The `fn.readers.file()` returns label IDs,
            #     : not label name directly extracted from directory path.
            #     : For example, cls_0000000 -> 0.
            img_files, labels = fn.readers.file(
                file_root=parentdir, random_shuffle=shuffle, name="FilesReader"
            )
            images = fn.decoders.image(img_files, device="mixed")
            images = fn.normalize(images, device="gpu")
            return images, labels.gpu()

        dali_pipeline = _build_pipeline(batch_size=batch_size, device_id=0)

        # Make dataset with DALIDataset.
        shapes = ((batch_size, 224, 224, 3), (batch_size,))
        dtypes = (tf.float32, tf.int32)
        dataset = dali_tf.DALIDataset(
            pipeline=dali_pipeline,
            batch_size=batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=0,
        )

        if return_n_classes:
            return dataset, n_data_size, n_classes
        else:
            return dataset, n_data_size

    train_ds, n_train_ds, n_classes = prepare_dataset(
        args, args.batch_size, "train", return_n_classes=True
    )
    num_batches_per_epoch = math.ceil(n_train_ds / args.batch_size)
    if args.no_validation:
        val_ds = None
        num_val_batches_per_epoch = None
    else:
        val_ds, n_val_ds = prepare_dataset(
            args, args.batch_size, "val", shuffle=False
        )
        num_val_batches_per_epoch = math.ceil(n_val_ds / args.batch_size)

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

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=opt
    )

    callbacks = []

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

        def on_epoch_begin(self, epoch, logs=None):
            self.starttime = timer()

        def on_epoch_end(self, epoch, logs=None):
            time = timer() - self.starttime
            img_sec = args.batch_size * num_batches_per_epoch / time
            print("Iter #%d: %.1f img/sec per %s" % (epoch, img_sec, device))
            # skip warm up epoch
            if epoch > 0:
                self.img_secs.append(img_sec)

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
        verbose=1,
    )

    # Save model.
    model.save(args.output_path)


if __name__ == "__main__":
    main()
