# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


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
    # NOTE: Assuming that this process can use all GPUs on this machine.
    n_total_gpus = len(gpus)

    strategy = tf.distribute.MirroredStrategy(
        devices=[f"/GPU:{idx}" for idx in range(n_total_gpus)]
    )

    # Prepare dataset from randomly generated files.
    global_batch_size = args.batch_size * n_total_gpus
    train_ds, n_train_ds, n_classes = prepare_dataset(
        args, args.batch_size, "train", strategy, return_n_classes=True
    )
    steps_per_epoch = math.ceil(n_train_ds / global_batch_size)
    if args.no_validation:
        val_ds = None
        validation_steps = None
    else:
        val_ds, n_val_ds = prepare_dataset(
            args, args.batch_size, "val", strategy, shuffle=False
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
    args, batch_size, subdir, strategy, return_n_classes=False, shuffle=True
):
    parentdir = os.path.join(args.input_path, subdir)
    if return_n_classes:
        n_classes = len(glob.glob(os.path.join(parentdir, "cls_*")))
    n_data = len(glob.glob(os.path.join(parentdir, "cls_*", "*.jpg")))

    # Build DALI data loading pipeline.
    # This pipeline will do: 1) reading file, 2) decoding jpeg,
    # 3) normalizing values, and 4) transferring data from CPU to GPU.
    @pipeline_def
    def _build_pipeline(shard_id, num_shards):
        # NOTE:
        # The `fn.readers.file()` returns label IDs,
        # not label name directly extracted from directory path.
        # For example, cls_0000000 -> 0.
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

    def dataset_fn(input_context):
        device_id = input_context.input_pipeline_id
        with tf.device(f"/gpu:{device_id}"):
            dali_pipeline = _build_pipeline(
                batch_size=batch_size,
                device_id=device_id,
                shard_id=device_id,
                num_shards=input_context.num_replicas_in_sync,
            )
            dataset = dali_tf.DALIDataset(
                pipeline=dali_pipeline,
                batch_size=batch_size,
                output_shapes=shapes,
                output_dtypes=dtypes,
                device_id=device_id,
            )
            return dataset

    # Make dataset distributed.
    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device=True,
        experimental_prefetch_to_device=False,  # TF2.4 or earlier.
        # experimental_fetch_to_device=False,  # TF2.5+
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA,  # noqa: E501
    )
    dataset = strategy.distribute_datasets_from_function(
        dataset_fn, input_options
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
