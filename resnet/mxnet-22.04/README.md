# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.
This repository is taken from NVIDIA's resnet50 benchmarking repository and is modified to be ran on top of Backend.AI.

## Requirements
* [MXNet 22.04-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# 2. Directions

## Dataset (tiny-imagenet)
The repository already contains the preprocessed `tiny-imagenet` in `data/preprocessed` folder. Use it to check whether all the benchmarking code functions properly. To run the actual benchmark, however, one has to download and preprocess the full size ImageNet which can be done by following the steps below.

1. Clone the public DeepLearningExamples repository
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/MxNet/Classification/RN50v1.5
git checkout 81ee705868a11d6fe18c12d237abe4a08aab5fd6
```

2. Build the ResNet50 MXNet NGC container
```
docker build . -t nvidia_rn50_mx
```

3. Start an interactive session in the NGC container to run preprocessing
```
nvidia-docker run --rm -it --ipc=host -v <path/to/store/raw/&/processed/data>:/data nvidia_rn50_mx
```

4. Download and unpack the data
* Download **Training images (Task 1 &amp; 2)** and **Validation images (all tasks)** at http://image-net.org/challenges/LSVRC/2012/2012-downloads (require an account)
* Extract the training data:
    ```
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ```
    
* Extract the validation data:
    ```
    mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val 
    tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    ```

5. Preprocess the dataset
```
./scripts/prepare_imagenet.sh <path/to/raw/imagenet> <path/to/save/preprocessed/data>
```

## Running Single-node
Running single-node benchmark is quite simple.

First, build the image
```
docker build -t mlperf-nvidia:image_classification .
```

And then, run
```
source run_singlenode.sh
```

## Running Multi-node
*Please Note that uploaded code is NOT properly tested for multi-node, hence requires caution to those attempting to run the multinode benchmark*
`run_multinode.sh` populates multiple docker containers using `image_classification` in a same docker network and tries to emulate the multi-node configuration.
First, build the image:
```
docker build -t mlperf-nvidia:image_classification .
```

Then, create a new network in docker:
```
docker network create multinode
```

And then, run
```
CONT=<name of the container> \
LOGDIR=<path/to/logging/directory> \
DATADIR=<path/to/the/dataset> \
NODENUM=<number of nodes> \
GPUPERNODE=<number of gpus per node> \
NETWORK=<name of the docker network where the containers are connected to> \
KEYREPO=<path/to/the/shared/ssh/key/folder> \
./run_multinode.sh
```
if you wish to run out-of-the-box, run the following:
```
CONT="mlperf-nvidia:image_classification" \
LOGDIR="./log" \
DATADIR="./data/preprocessed" \
NODENUM="2" \
GPUPERNODE="1" \
NETWORK="multinode" \
KEYREPO="./key_repo" \
./run_multinode.sh
```

`run_multinode.sh` handles the ssh authentication configurations to enable `horovodrun` passwordless-ssh between each docker containers. It does so by mounting `./key_repo` to the docker container and exchaning public keys in that folder. Concretely, one must ensure that `./key_repo/authorized_keys` and `./key_repo/known_hosts` files exist.