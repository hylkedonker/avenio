#!/bin/bash
docker run \
    -v $(pwd)/models/:/package/models/ \
    local/avenio \
    python3 survival.py
