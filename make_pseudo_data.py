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
import os

from PIL import Image
import numpy as np


def make_and_save_images(n_images, num_classes, outdir, filename_prefix):
    class_dirpath_list = []
    print("try to create class directories.")
    for c in range(num_classes):
        class_dirpath = os.path.join(
            outdir, filename_prefix, f"cls_{c:03d}"
        )
        print(f"\t class[{c}]: path={class_dirpath}")
        os.makedirs(class_dirpath)
        class_dirpath_list.append(class_dirpath)

    imgids_list = np.array_split(np.arange(n_images), len(class_dirpath_list))
    for cid, class_dirpath in enumerate(class_dirpath_list):
        for imgid in imgids_list[cid]:
            filepath = os.path.join(
                class_dirpath, f"{filename_prefix}_{imgid:07d}.jpg"
            )
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
        required=True,
        type=int,
        help=(
            "Number of classes."
            "Directories for each class will be created "
            "and each image file will be stored into each class directory."
        ),
    )
    parser.add_argument(
        "--outdir", required=True, type=str, help="Output dir."
    )
    parser.add_argument(
        "--val-ratio",
        required=True,
        type=float,
        help=(
            "Validation data ratio: "
            "--num-images * --val-ratio = #val_images."
        ),
    )
    args = parser.parse_args()
    print(args)

    n_val_images = int(args.num_images * args.val_ratio)
    n_train_images = args.num_images - n_val_images
    print(
        f"#images {args.num_images} -> "
        f"(#train, #val) = ({n_train_images}, {n_val_images})"
    )

    make_and_save_images(
        n_train_images, args.num_classes, args.outdir, "train"
    )
    make_and_save_images(n_val_images, args.num_classes, args.outdir, "val")
    print("done.")


if __name__ == "__main__":
    main()
