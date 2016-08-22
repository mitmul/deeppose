#! /bin/bash
# Copyright (c) 2016 Shunta Saito

lhs=$1
rhs=$2
convert $1 ${lhs%.*}.gif
convert $2 ${rhs%.*}.gif
convert -delay 100 -loop 0 ${lhs%.*}.gif ${rhs%.*}.gif -layers OptimizeTransparency $3
