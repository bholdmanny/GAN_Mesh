#!/bin/bash

SelDir()
{
    src=$1; dst=$2; beg=$3; end=$4
    for i in $(ls $src);
    do
        #if [ $i -ge $beg ]&&[ $i -le $end ]; then echo "$i" >> List; fi
        if [ $i -ge $beg ]&&[ $i -le $end ]; then cp -r $src/$i $dst; fi
    done
}#0-263,268-473

src=/home/hua.fu/Tools/CASIA-WebFace
SelDir $src AI_Train_1 0 263
SelDir $src AI_Train_2 268 473