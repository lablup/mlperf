> key_repo/authorized_keys
> key_repo/known_hosts

docker stop $(docker ps -aqf "name=container-1")
docker stop $(docker ps -aqf "name=container-2")
