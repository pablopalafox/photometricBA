#!/usr/bin/env bash

set -x
set -e

DLNAME=V1_01_easy
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/$DLNAME/$DLNAME.zip
unzip -d $DLNAME $DLNAME.zip
rm $DLNAME.zip
rm -rf $DLNAME/__MACOSX
chmod -R go-w $DLNAME
