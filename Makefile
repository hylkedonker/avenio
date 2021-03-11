BUILD=docker build
USER=umcg-hcdonker
CLUSTER_DIR=/groups/umcg-griac/tmp01/rawdata/${USER}/avenio/singularity
TMP_DIR=${HOME}/tmp
BRANCH_NAME=$(shell git branch | grep '*' | cut -d' ' -f2 | sed -E 's/feature\///')
DOCKER_IMAGE=avenio_${BRANCH_NAME}.tar

all: singularity

docker:
	$(BUILD) -f Dockerfile -t avenio/${BRANCH_NAME} .

singularity: docker
	docker save avenio/${BRANCH_NAME} -o ${TMP_DIR}/${DOCKER_IMAGE}
	rsync -av ${TMP_DIR}/${DOCKER_IMAGE} ${USER}@airlock+gearshift:${CLUSTER_DIR}
	rm ${TMP_DIR}/${DOCKER_IMAGE}
	ssh ${USER}@airlock+gearshift \
		"export SINGULARITY_CACHEDIR=${CLUSTER_DIR}/cache; \
		singularity build \
		--force \
		--sandbox ${CLUSTER_DIR}/${BRANCH_NAME} \
		docker-archive://${CLUSTER_DIR}/${DOCKER_IMAGE}"
