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
import json
import math
import os

import tensorflow as tf


def main():
    args = parse_args()

    # Setup worker information.
    assert -1 < args.worker_id < len(args.worker_addrs)
    _worker_info = []
    for addr in args.worker_addrs:
        addr_port = f"{addr}:{args.worker_base_port}"
        _worker_info.append(addr_port)
    _task_info = {"type": "worker", "index": args.worker_id}
    tf_config_obj = {"cluster": {"worker": _worker_info}, "task": _task_info}
    print(f"TF_CONFIG: {tf_config_obj}")
    os.environ["TF_CONFIG"] = json.dumps(tf_config_obj)

    # Setup all GPUs.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # NOTE: Assuming that each worker has the same number of GPUs.
    n_total_gpus = len(gpus) * len(args.worker_addrs)

    # NOTE:
    # If you are interested in only single-node multi-GPUs training,
    # please use MirroredStrategy instead of MultiWorkerMirroredStrategy.
    # https://www.tensorflow.org/guide/distributed_training#mirroredstrategy
    # NOTE:
    # Need to make a strategy instance before CollectiveOp.
    # This means that we also need to put the code for data preparation after
    # making strategy.
    # Please see the following note for more details.
    # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    # NOTE:
    # If you are interested in the performance improvement for
    # communications between nodes, following page might be helpful.
    # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#performance
    # NOTE:
    # When NCCL is explicitly set as communication method,
    # Segmentation fault sometimes happens. So, the AUTO mode is used here.
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CollectiveCommunication.AUTO  # noqa: E501
        )
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
        # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#train_the_model
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
        verbose=(args.worker_id == 0),
    )

    # Save model into files.
    # See the link below for more details.
    # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#model_saving_and_loading
    def _is_chief(task_type, task_id):
        # If `task_type` is None, this may be operating as single worker,
        # which works effectively as chief.
        return (
            task_type is None
            or task_type == "chief"
            or (task_type == "worker" and task_id == 0)
        )

    def _get_temp_dir(dirpath, task_id):
        base_dirpath = "workertemp_" + str(task_id)
        temp_dir = os.path.join(dirpath, base_dirpath)
        tf.io.gfile.makedirs(temp_dir)
        return temp_dir

    def write_filepath(filepath, task_type, task_id):
        if _is_chief(task_type, task_id):
            return filepath

        if os.path.isfile(filepath):
            dirpath = os.path.dirname(filepath)
        else:
            dirpath = filepath
        return os.path.join(
            _get_temp_dir(dirpath, task_id), os.path.basename(filepath)
        )

    task_type, task_id = (
        strategy.cluster_resolver.task_type,
        strategy.cluster_resolver.task_id,
    )
    write_model_path = write_filepath(args.output_path, task_type, task_id)
    model.save(write_model_path)
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
        description="TensorFlow2-Keras MultiWorker Example",
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

    parser.add_argument(
        "--worker-addrs",
        type=str,
        required=True,
        nargs="+",
        help="Worker node addresses.",
    )
    parser.add_argument(
        "--worker-base-port",
        type=int,
        default=12345,
        help="Base port number for each worker node.",
    )
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID.")

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main()
