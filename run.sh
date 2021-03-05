#!/bin/bash
docker run \
    -ti \
    -v /home/donkerhc/aveniodata/BamGraz/:/data/bam/:ro \
    -v $(pwd):/package/ \
    local/avenio \
    bash
