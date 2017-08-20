#!/bin/bash
SCRIPT="flows/$1.sh"

bash $SCRIPT ${@:2} &
tail -f results/training.log
