#!/bin/bash
SCRIPT="flows/$1.sh"

exec bash $SCRIPT ${@:2} > /tmp/siftbow.log 2>&1 &

