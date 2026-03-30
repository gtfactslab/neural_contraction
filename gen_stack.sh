#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)/outputs"

ffmpeg -i "$DIR/hover/video.mp4" \
       -i "$DIR/spiral/video.mp4" \
       -i "$DIR/figure_eight/video.mp4" \
       -i "$DIR/trefoil/video.mp4" \
  -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \
  -map "[v]" -c:v libx264 -crf 18 "$DIR/combined.mp4"

echo "Saved to $DIR/combined.mp4"
