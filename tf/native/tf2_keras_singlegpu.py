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
import math
import os

import tensorflow as tf


def main():
    args = parse_args()

    # Setup all GPUs.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Prepare dataset from randomly generated files.
    train_ds, n_train_ds, n_classes = prepare_dataset(
        args, args.batch_size, 'train', return_n_classes=True)
    val_ds, n_val_ds = prepare_dataset(
        args, args.batch_size, 'val', shuffle=False)
    steps_per_epoch = math.ceil(n_train_ds / args.batch_size)
    validation_steps = math.ceil(n_val_ds / args.batch_size)

    # Setup model, etc.
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    opt = tf.keras.optimizers.RMSprop()
    model = build_model(n_classes)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # Start training.
    model.fit(train_ds,
              epochs=args.num_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_ds,
              validation_steps=validation_steps,
              verbose=1)
    # NOTE: You must use one another dataset like test_ds
    #     : for the actual last evlauation.
    test_scores = model.evaluate(val_ds, steps=validation_steps, verbose=2)
    print(f'Test loss: {test_scores[0]}')
    print(f'Test accuracy: {test_scores[1]}')

    # Save model into files.
    model.save(args.output_path)
    print('done.')


def prepare_dataset(args, batch_size, subdir, return_n_classes=False, shuffle=True):
    parentdir = os.path.join(args.input_path, subdir)
    if return_n_classes:
        import glob
        n_classes = len(glob.glob(os.path.join(parentdir, 'cls_*')))

    # Load dataset from randomly generated files.
    list_ds = tf.data.Dataset.list_files(
        os.path.join(parentdir, 'cls_*', '*.jpg'),
        shuffle=shuffle)
    n_data = len(list_ds)

    list_ds = list_ds.repeat()
    if shuffle:
        list_ds = list_ds.shuffle(buffer_size=batch_size*2)
    def process_path(file_path):
        # Assuming that the structure of file_path like below.
        #   "/path/to/parentdir/subdir/cls_${class_id}/[train|val]_${imgno}.jpg".
        label = tf.strings.split(file_path, os.sep)[-2]
        label = tf.strings.to_number(tf.strings.split(label, '_')[-1], tf.int32)
        image = tf.io.decode_jpeg(tf.io.read_file(file_path))
        return image, label
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    def normalization(image, label):
        # Applying simple normalization.
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        return image, label
    dataset = labeled_ds.batch(batch_size)
    dataset = dataset.map(normalization, num_parallel_calls=tf.data.AUTOTUNE)

    if return_n_classes:
        return dataset, n_data, n_classes
    else:
        return dataset, n_data

def build_model(n_classes):
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model

def parse_args():
    parser = argparse.ArgumentParser(
        description='TensorFlow2-Keras single GPU Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-path', type=str, default='./images',
                        help='a parent directory path to input image files')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size')

    parser.add_argument('--num-epochs', type=int, default=10,
                        help='number of epochs')

    parser.add_argument('--output-path', type=str, default='./models',
                        help='output path to store saved model')

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    main()
