#!/bin/bash

# Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/raid/datasets/train-val-recordio-passthrough}"
: "${LOGDIR:=$(pwd)/results}"
: "${COPY_DATASET:=}"

echo $COPY_DATASET

if [ ! -z $COPY_DATASET ]; then
    readonly copy_datadir=$COPY_DATASET
    mkdir -p "${DATADIR}"
    ${CODEDIR}/copy-data.sh "${copy_datadir}" "${DATADIR}"
    ls ${DATADIR}
fi

# Other vars
readonly _seed_override=${SEED:-}
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=image_classification
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${LOGDIR}:/results")

# MLPerf vars
MLPERF_HOST_OS=$(
    source /etc/os-release
    source /etc/dgx-release || true
    echo "${PRETTY_NAME} / ${DGX_PRETTY_NAME:-???} ${DGX_OTA_VERSION:-${DGX_SWBUILD_VERSION:-???}}"
)
export MLPERF_HOST_OS

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(MLPERF_HOST_OS SEED)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
nvidia-docker run --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
#make sure container has time to finish initialization
sleep 30
docker exec -it "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Print system info
        docker exec -it "${_cont_name}" python -c "
import mlperf_log_utils
from mlperf_logging.mllog import constants

mlperf_log_utils.mlperf_submission_log(constants.RESNET)"

        # Clear caches
#         if [ "${CLEAR_CACHES}" -eq 1 ]; then
#             sync && sudo /sbin/sysctl vm.drop_caches=3
#             docker exec -it "${_cont_name}" python -c "
# import mlperf_log_utils
# from mlperf_logging.mllog import constants

# mlperf_log_utils.mx_resnet_print_event(key=constants.CACHE_CLEAR, val=True)"
#         fi

        echo "printing out the run command"
        echo "docker exec -it "${_config_env[@]}" "${_cont_name}" \
	       mpirun --allow-run-as-root --bind-to none --np ${DGXNGPU} ./run_and_time.sh"

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
        docker exec -it "${_config_env[@]}" "${_cont_name}" \
	       mpirun --allow-run-as-root --bind-to none --np ${DGXNGPU} ./run_and_time.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done

# docker exec -it --env=BATCHSIZE --env=DALI_CACHE_SIZE --env=DALI_CROP_BUFFER_HINT --env=DALI_DECODER_BUFFER_HINT --env=DALI_DONT_USE_MMAP --env=DALI_HW_DECODER_LOAD --env=DALI_NORMALIZE_BUFFER_HINT --env=DALI_NVJPEG_MEMPADDING --env=DALI_PREALLOCATE_HEIGHT --env=DALI_PREALLOCATE_WIDTH --env=DALI_PREFETCH_QUEUE --env=DALI_ROI_DECODE --env=DALI_THREADS --env=DALI_TMP_BUFFER_HINT --env=DGXHT --env=DGXNGPU --env=DGXNNODES --env=DGXNSOCKET --env=DGXSOCKETCORES --env=DGXSYSTEM --env=EVAL_OFFSET --env=EVAL_PERIOD --env=HOROVOD_CYCLE_TIME --env=HOROVOD_FUSION_THRESHOLD --env=HOROVOD_NUM_NCCL_STREAMS --env=KVSTORE --env=LABELSMOOTHING --env=LARSETA --env=LR --env=LRSCHED --env=MOM --env=MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_BWD --env=MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_FWD --env=MXNET_CUDNN_NHWC_BN_HEURISTIC_BWD --env=MXNET_CUDNN_NHWC_BN_HEURISTIC_FWD --env=MXNET_CUDNN_WARN_ON_IGNORED_FLAGS --env=MXNET_ENABLE_CUDA_GRAPHS --env=MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD --env=MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD --env=MXNET_EXTENDED_NORMCONV_SUPPORT --env=MXNET_HOROVOD_NUM_GROUPS --env=MXNET_OPTIMIZER_AGGREGATION_SIZE --env=NETWORK --env=NUMEPOCHS --env=OMPI_MCA_btl --env=OPTIMIZER --env=WALLTIME --env=WARMUP_EPOCHS --env=WD --env=MLPERF_HOST_OS --env=SEED image_classification mpirun --allow-run-as-root --bind-to none --np 2 ./run_and_time.sh
