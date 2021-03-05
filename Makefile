BUILD=docker build
all:
	$(BUILD) -f Dockerfile -t local/avenio .
