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
import math
import os

import tensorflow as tf

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.plugin.tf as dali_tf


def main():
    args = parse_args()

    # Setup all GPUs.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Prepare dataset from randomly generated files.
    train_ds, n_train_ds, n_classes = prepare_dataset(
        args, args.batch_size, "train", return_n_classes=True
    )
    steps_per_epoch = math.ceil(n_train_ds / args.batch_size)
    if args.no_validation:
        val_ds = None
        validation_steps = None
    else:
        val_ds, n_val_ds = prepare_dataset(
            args, args.batch_size, "val", shuffle=False
        )
        validation_steps = math.ceil(n_val_ds / args.batch_size)

    # Setup model, etc.
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)

    opt = tf.keras.optimizers.SGD(learning_rate=args.lr)
    model = build_model(n_classes)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )

    # Start training.
    model.fit(
        train_ds,
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        verbose=1,
    )

    # Save model into files.
    model.save(args.output_path)
    print("done.")


def prepare_dataset(
    args, batch_size, subdir, return_n_classes=False, shuffle=True
):
    parentdir = os.path.join(args.input_path, subdir)
    if return_n_classes:
        n_classes = len(glob.glob(os.path.join(parentdir, "cls_*")))
    n_data = len(glob.glob(os.path.join(parentdir, "cls_*", "*.jpg")))

    # Build DALI data loading pipeline.
    # This pipeline will do: 1) reading file, 2) decoding jpeg,
    # 3) normalizing values, and 4) transferring data from CPU to GPU.
    @pipeline_def
    def _build_pipeline():
        # NOTE:
        # The `fn.readers.file()` returns label IDs,
        # not label name directly extracted from directory path.
        # For example, cls_0000000 -> 0.
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
        return dataset, n_data, n_classes
    else:
        return dataset, n_data


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorFlow2-Keras single GPU Example",
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
    print(args)

    return args


if __name__ == "__main__":
    main()
