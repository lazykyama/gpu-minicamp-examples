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
import os

from PIL import Image
import numpy as np


def make_and_save_images(n_images, num_classes, outdir, filename_prefix):
    if num_classes is None:
        class_dirpath_list = [outdir]
    else:
        class_dirpath_list = []
        print("try to create class directories.")
        for c in range(num_classes):
            class_dirpath = os.path.join(outdir, filename_prefix, f"cls_{c:03d}")
            print(f"\t class[{c}]: path={class_dirpath}")
            os.makedirs(class_dirpath)
            class_dirpath_list.append(class_dirpath)

    imgids_list = np.array_split(np.arange(n_images), len(class_dirpath_list))
    for cid, class_dirpath in enumerate(class_dirpath_list):
        for imgid in imgids_list[cid]:
            filepath = os.path.join(class_dirpath, f"{filename_prefix}_{imgid:07d}.jpg")
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
            )
            img.save(filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-images", required=True, type=int, help="Number of images."
    )
    parser.add_argument(
        "--num-classes",
        default=None,
        type=int,
        help=(
            "Number of classes."
            "If specified, directories for each class will be created "
            "and each image file will be stored into each class directory."
            "If not specified, all files will be stored into one directory."
        ),
    )
    parser.add_argument("--outdir", required=True, type=str, help="Output dir.")
    parser.add_argument(
        "--val-ratio",
        required=True,
        type=float,
        help="Validation data ratio: --num-images * --val-ratio = #val_images.",
    )
    args = parser.parse_args()
    print(args)

    n_val_images = int(args.num_images * args.val_ratio)
    n_train_images = args.num_images - n_val_images
    print(
        f"#images {args.num_images} -> (#train, #val) = ({n_train_images}, {n_val_images})"
    )

    make_and_save_images(n_train_images, args.num_classes, args.outdir, "train")
    make_and_save_images(n_val_images, args.num_classes, args.outdir, "val")
    print("done.")


if __name__ == "__main__":
    main()
