echo "config_DGXA100.sh RUNNING"

source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

## DL params
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="4"
export KVSTORE="horovod"
export LR="10.5"
export WARMUP_EPOCHS="2"
export EVAL_OFFSET="2" # Targeting epoch no. 35
export EVAL_PERIOD="4"
export WD="5.0e-05"
export MOM="0.9"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS=${NUMEPOCHS:-"37"}

export NETWORK="resnet-v1b-stats-fl"
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_BWD=0
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_FWD=0

export DALI_THREADS="6"
export DALI_PREFETCH_QUEUE="5"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_HW_DECODER_LOAD="0"
export DALI_CACHE_SIZE="0"
export DALI_ROI_DECODE="1"

#DALI buffer presizing hints
export DALI_PREALLOCATE_WIDTH="5980"
export DALI_PREALLOCATE_HEIGHT="6430"
export DALI_DECODER_BUFFER_HINT="1315942" #1196311*1.1
export DALI_CROP_BUFFER_HINT="165581" #150528*1.1
export DALI_TMP_BUFFER_HINT="355568328" #871491*batch_size
export DALI_NORMALIZE_BUFFER_HINT="441549" #401408*1.1

export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=2
export MXNET_HOROVOD_NUM_GROUPS=1
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:40:00

export NAME=Ubuntu
export VERSION='18.04.5 LTS (Bionic Beaver)'
export ID=ubuntu
export ID_LIKE=debian
export PRETTY_NAME='Ubuntu 18.04.5 LTS'
export VERSION_ID=18.04
export HOME_URL=https://www.ubuntu.com/
export SUPPORT_URL=https://help.ubuntu.com/
export BUG_REPORT_URL=https://bugs.launchpad.net/ubuntu/
export PRIVACY_POLICY_URL=https://www.ubuntu.com/legal/terms-and-policies/privacy-policy
export VERSION_CODENAME=bionic
export UBUNTU_CODENAME=bionic

export MLPERF_HOST_OS=$(
    source /etc/os-release
    source /etc/dgx-release || true
)
export SEED=${_seed_override:-$RANDOM}

