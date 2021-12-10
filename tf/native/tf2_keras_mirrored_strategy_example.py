# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math
import os

import tensorflow as tf


def main():
    args = parse_args()

    # Setup all GPUs.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # NOTE: Assuming that this process can use all GPUs on this machine.
    n_total_gpus = len(gpus)

    strategy = tf.distribute.MirroredStrategy(
        devices=[f"/GPU:{idx}" for idx in range(n_total_gpus)]
    )

    # Prepare dataset from randomly generated files.
    global_batch_size = args.batch_size * n_total_gpus
    train_ds, n_train_ds, n_classes = prepare_dataset(
        args, global_batch_size, "train", return_n_classes=True
    )
    steps_per_epoch = math.ceil(n_train_ds / global_batch_size)
    if args.no_validation:
        val_ds = None
        validation_steps = None
    else:
        val_ds, n_val_ds = prepare_dataset(
            args, global_batch_size, "val", shuffle=False
        )
        validation_steps = math.ceil(n_val_ds / global_batch_size)

    # Setup model, etc.
    with strategy.scope():
        # NOTE:
        # According to the tutorial below,
        # model building/compiling need to be within `strategy.scope()`.
        # https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_keras_modelfit
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
    args, global_batch_size, subdir, return_n_classes=False, shuffle=True
):
    parentdir = os.path.join(args.input_path, subdir)
    if return_n_classes:
        import glob

        n_classes = len(glob.glob(os.path.join(parentdir, "cls_*")))

    # Load dataset from randomly generated files.
    list_ds = tf.data.Dataset.list_files(
        os.path.join(parentdir, "cls_*", "*.jpg"), shuffle=shuffle
    )
    n_data = len(list_ds)

    list_ds = list_ds.repeat()
    if shuffle:
        list_ds = list_ds.shuffle(buffer_size=global_batch_size * 2)

    def process_path(file_path):
        # Assuming that the structure of file_path like below.
        #   "/path/to/parentdir/subdir/cls_${class_id}/[train|val]_${imgno}.jpg".
        label = tf.strings.split(file_path, os.sep)[-2]
        label = tf.strings.to_number(
            tf.strings.split(label, "_")[-1], tf.int32
        )
        image = tf.io.decode_jpeg(tf.io.read_file(file_path))
        return image, label

    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    def normalization(image, label):
        # Applying simple normalization.
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        return image, label

    dataset = labeled_ds.batch(global_batch_size)
    dataset = dataset.map(normalization, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Set DATA as a shard mode.
    # NOTE:
    # The reason why DATA is set as a shard mode is that
    # this example doesn't use file level sharded dataset.
    # Please read the doc below for more details.
    # https://www.tensorflow.org/tutorials/distribute/input#sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    dataset = dataset.with_options(options)

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
        description="TensorFlow2-Keras MirroredStrategy Example",
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
