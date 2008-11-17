#!/bin/sh

echo
echo "Libraries required to build this project:"
echo "  * Boost (www.boost.org)"
echo "  * ImageMagick + Magick++ (http://www.imagemagick.org/)"
echo 

cd cell
make && echo "Start ./trace_spu to execute rendering performance test"
