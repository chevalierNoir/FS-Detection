#! /bin/bash

pip install -r requirements.txt --no-cache-dir
pip install warpctc-pytorch==0.2.2+torch11.cpu -f https://github.com/espnet/warp-ctc/releases/tag/v0.2.2 --no-cache_dir
