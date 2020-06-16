#!/bin/bash
docker run \
    -ti \
    -v /home/donkerhc/aveniodata/BamGraz:/data/bam:ro \
    -v /home/donkerhc/workspace/jupyter/avenio/read_count:/package \
    local/avenio \
    bash
