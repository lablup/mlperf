#! /bin/bash

set -euxo pipefail

: "${DATADIR:?DATADIR is not set}"
: "${LOGDIR:?LOGDIR is not set}"
: "${CONT:?CONT is not set}"
: "${NODENUM:?NODENUM is not set}"
: "${GPUPERNODE:?GPUPERNODE is not set}"
: "${NETWORK:?NETWORK is not set}"
: "${KEYREPO:?KEYREPO is not set}"

IP_PARAMS=""

for _container_num in $(seq 1 "${NODENUM}"); do
	 nvidia-docker run --init --rm --detach \
	 				--net=${NETWORK} \
					--uts=host \
					--ipc=host \
					--security-opt=seccomp=unconfined \
					--ulimit=stack=67108864 \
					--ulimit=memlock=-1 \
					--name=container-${_container_num} \
					--gpus=${GPUPERNODE} \
					--volume=${DATADIR}:/data \
					--volume=${LOGDIR}:/results \
					--volume=${KEYREPO}:/key_repo \
					${CONT} sleep infinity
					
	docker exec container-${_container_num} bash -c 'service ssh start'
	docker exec container-${_container_num} bash -c "ssh-keygen -f /root/.ssh/id_rsa -q -N ''; cat /root/.ssh/id_rsa.pub >> /key_repo/authorized_keys"
	docker exec container-${_container_num} bash -c "ssh-keyscan -t rsa container-${_container_num} >> /key_repo/known_hosts"
	docker exec container-${_container_num} touch /container-${_container_num}
	 
	CONTID=$(docker ps -aqf "name=container-${_container_num}")
	IP_PARAMS+=$(echo $(docker inspect ${CONTID} -f "{{json .NetworkSettings.Networks.multinode.IPAddress}}"):${GPUPERNODE}, | tr -d '"')
done

for _container_num in $(seq 1 "${NODENUM}"); do
	docker exec container-${_container_num} bash -c "cat /key_repo/authorized_keys >> /root/.ssh/authorized_keys"
	docker exec container-${_container_num} bash -c "cat /key_repo/known_hosts >> /root/.ssh/known_hosts"
done

# run multinode benchmark
IP_PARAMS=$(echo ${IP_PARAMS} | sed "s/,$//")
docker exec container-1 bash -c "chmod +x config_DGXA100.sh; source config_DGXA100.sh; horovodrun -H ${IP_PARAMS} -np $((${GPUPERNODE}*${NODENUM})) run_and_time.sh"