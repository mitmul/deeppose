#! /bin/bash

convert $1 ${$1%.*}.gif
convert $2 ${$2%.*}.gif
convert -delay 100 -loop 0 ${$1%.*}.gif ${$2%.*}.gif $3
