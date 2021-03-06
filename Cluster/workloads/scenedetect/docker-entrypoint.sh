#!/bin/bash
if [[ -z "$NUM_REDIS" ]] || [[ -z "$SELF_PORT" ]] || [[ -z "$SELF_HOST" ]] || [[ $NUM_REDIS -le 0 ]]; then
    ./aiohttp-scenedetect.py
else
    REDIS_HOST=$SELF_HOST
    REDIS_PORT=$((6381+$((${SELF_PORT} % ${NUM_REDIS}))))
    ./aiohttp-scenedetect.py
fi