#!/bin/bash

CWD=$(pwd)
# clone espent and checkout to specific version
# git clone https://github.com/espnet/espnet.git
# cd ${CWD}/espnet
# git reset --hard 9605c61c6dcef71bb8ab54343c186b3c98879712
# cd ${CWD}

# install espnet yourself following the installation instructions

# copy model files to espnet
cp zeroshot_langemb/espnet/asr/pytorch_backend/recog.py espnet/espnet/asr/pytorch_backend/
cp zeroshot_langemb/espnet/bin/asr_recog.py zeroshot_langemb/espnet/bin/asr_train.py espnet/espnet/bin/
cp zeroshot_langemb/espnet/nets/pytorch_backend/*.py espnet/espnet/nets/pytorch_backend/

