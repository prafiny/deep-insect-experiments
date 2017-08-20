#!/bin/bash
SCRIPT="flows/$1.sh"

exec bash $SCRIPT ${@:2} > /tmp/convnets.log 2>&1 &
