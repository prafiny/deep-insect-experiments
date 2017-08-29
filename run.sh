#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR
exit
#HOST=gpu
DISTANT_ROOTPATH=/home/maxime/deploy
LOCAL_ROOTPATH="$DIR"

if [ $# -lt 1 ]; then
    echo "USAGE : ./run.sh FOLDER [PARAMS…]"
    exit
fi

folder="$1"
#bname_folder=$( realpath --relative-to=$LOCAL_ROOTPATH $folder )
bname_folder=$( readlink -f "$folder" )
bname_folder=${bname_folder##*/}
tim=$(date +"%F_%H-%M-%S")
distant_folder="$DISTANT_ROOTPATH/${tim}_${bname_folder}"

params="${@:2}"

#ssh $HOST "[ -d $distant_folder ] && rm -r \"$distant_folder\" || mkdir -p \"$distant_folder\""
#scp -rp "$folder" $HOST:"$distant_folder"
cp -r "$folder" "$distant_folder"

if [ -f "$folder/deps.sh" ]; then
    source "$folder/deps.sh"
    for d in ${!deps[@]}; do
        cp -r "$LOCAL_ROOTPATH/$d" "$distant_folder/${deps[$d]}"
	#scp -rp "$LOCAL_ROOTPATH/$d" $HOST:"$distant_folder/${deps[$d]}"
    done
fi

cd "$distant_folder"
bash ./post_deploy.sh $params

