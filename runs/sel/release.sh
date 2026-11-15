#!/bin/bash
until [ "$(TZ=America/New_York date +%H%M)" -ge 1738 ]; do sleep 30; done
docker ps --format "{{.Names}}" | grep -E "^rtm-d4" | xargs -r docker rm -f
sleep 20; docker start dsv41-agent-candidate-v9 && echo DSSTARTED
