#!/bin/bash
docker run \
    -ti \
    -v /home/donkerhc/aveniodata/BamGraz:/data/bam:ro \
    -v /home/donkerhc/workspace/jupyter/avenio/fragment_count:/package \
    local/avenio \
    bash
