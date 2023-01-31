#! /bin/bash

set -euxo pipefail

: "${CONT:? CONT is not set}"
: "${DATADIR:? DATADIR is not set}"
: "${LOGDIR:? LOGDIR is not set}"

nvidia-docker run --rm --init --detach \
    --net=host \
    --uts=host \
    --ipc=host \
    --security-opt=seccomp=unconfined     \
    --ulimit=stack=67108864 \
    --ulimit=memlock=-1     \
    --name=image_classification \
    --volume=${DATADIR}:/data \
    --volume=${LOGDIR}:/results   \
    -p 2022:22 \
    ${CONT} sleep infinity

docker exec image_classification bash -c "chmod +x config_DGXA100.sh; source config_DGXA100.sh; mpirun --allow-run-as-root --bind-to none --np 2 ./run_and_time.sh"

