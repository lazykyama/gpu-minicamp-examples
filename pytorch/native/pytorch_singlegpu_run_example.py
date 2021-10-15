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
import os
import time

import torch

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


def _check_pytorch_version():
    version_info = tuple(map(int, torch.__version__.split('.')[:2]))
    if version_info[0] < 1:
        # Not supported version because of old major version, 0.x.
        return False
    if version_info[1] < 9:
        # Not supported version because of old minor version, 1.8 or earlier.
        return False
    return True

def main():
    is_expected_version = _check_pytorch_version()
    if not is_expected_version:
        raise RuntimeError((
            "Your PyTorch version is not expected in this example code."
            "1.9+ is required."))

    args = parse_args()

    device = torch.device(
        f'cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare output directory.
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.isdir(args.output_path):
        raise RuntimeError(f'{args.output_path} exists, but it is not a directory.')

    trainloader, valloader, n_classes = prepare_dataset(
        args.input_path, args.batch_size)

    model = build_model(n_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # NOTE: If you are interested in the acceleration by Tensor Cores,
    #     : please read the following doc.
    # https://pytorch.org/docs/stable/amp.html
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")

        running_loss = 0.0
        # NOTE: This is a simplified way to measure the time.
        #     : You should use more precise method to know the performance.
        starttime = time.perf_counter()
        for i, data in enumerate(trainloader):
            # NOTE: If you want to overlap the data transferring between CPU-GPU,
            #     : you need to additionally implement custom dataloader,
            #     : or use pinned memory.
            #     : Following pages could help you.
            # https://github.com/NVIDIA/DeepLearningExamples/blob/f24917b3ee73763cfc888ceb7dbb9eb62343c81e/PyTorch/Classification/ConvNets/image_classification/dataloaders.py#L347
            # https://pytorch.org/docs/stable/data.html#memory-pinning
            inputs = data[0].to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)

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
                print((
                    f'\t [iter={i+1:05d}] training loss = '
                    f'{running_loss / (i+1):.3f}'))
        # End of each epoch.

        # Calculate validation result.
        running_valloss = 0.0
        n_valiter = len(valloader)
        model.eval()
        with torch.no_grad():
            for valdata in valloader:
                val_in = valdata[0].to(
                    device, non_blocking=True)
                val_label = valdata[1].to(
                    device, non_blocking=True)
                valout = model(val_in)
                valloss = criterion(valout, val_label)
                running_valloss += valloss.item()
        model.train()

        # Show this epoch time and training&validation losses.
        # NOTE: This time includes vaidation time.
        duration = time.perf_counter() - starttime
        print((
            f'\t [iter={i+1:05d}] '
            f'{duration:.3f}s {duration*1000. / i:.3f}ms/step, '
            f'loss = {running_loss / (i+1):.3f}, '
            f'val_loss = {running_valloss / n_valiter:.3f}'))

    # Save model.
    model_filepath = os.path.join(args.output_path, 'model.pth')
    torch.save(model.state_dict(), model_filepath)

    print('done.')


def prepare_dataset(datadir, batch_size):
    n_classes = len(glob.glob(
        os.path.join(datadir, 'train', 'cls_*')))

    # Prepare transform ops.
    # Basically, PIL object should be converted into tensor.
    transform = transforms.Compose([transforms.ToTensor()])

    # Prepare train dataset.
    # NOTE: ImageFolder assumes that `root` directory contains 
    #     : several class directories like below.
    # root/cls_000, root/cls_001, root/cls_002, ...
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(datadir, 'train'), transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=8)

    # Prepare val dataset.
    valset = torchvision.datasets.ImageFolder(
        root=os.path.join(datadir, 'val'), transform=transform)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size,
        shuffle=False, num_workers=8)

    print(f'trainset.size = {len(trainset)}')
    print(f'valset.size = {len(valset)}')

    return trainloader, valloader, n_classes

def build_model(n_classes):
    model = models.resnet50(pretrained=False)
    n_fc_in_feats = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(n_fc_in_feats, 512),
        torch.nn.Linear(512, n_classes)
    )
    return model

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch torch.distributed.run Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-path', type=str, default='./images',
                        help='a parent directory path to input image files')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size')

    parser.add_argument('--num-epochs', type=int, default=10,
                        help='number of epochs')

    parser.add_argument('--output-path', type=str, default='./models',
                        help='output path to store saved model')

    parser.add_argument('--logging-interval', type=int, default=10,
                        help='logging interval')

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    main()