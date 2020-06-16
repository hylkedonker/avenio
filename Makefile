DOCKER_BUILD=docker build
all:
	$(DOCKER_BUILD) -f Dockerfile -t local/avenio .
