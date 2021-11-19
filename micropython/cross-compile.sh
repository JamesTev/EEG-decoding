#!/bin/bash
find lib -maxdepth 1 -type f -exec mpy-cross {} \;
mv lib/*.mpy mpy-modules/
