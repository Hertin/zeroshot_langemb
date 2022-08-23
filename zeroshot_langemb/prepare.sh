#!/bin/bash

# Copyright 2020 Johns Hopkins University (Piotr Żelasko)
# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# rnnlm related
use_lm=false
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
# tag="train_li10_ctc_c" # tag for managing experiments.
ita=0
tag="dptr_add_el6_eu512" # tag for managing experiments.
train_config=conf/train_li10_dptr_add_el6_eu512.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode_li10_dptr_nosignature.yaml

exp_tag="cross-lingual"


# Generate configs with local/prepare_experiment_configs.py
langs_config=
remove_lang=
retain_lang="307,103,101,402,107,206"
if [ $langs_config ]; then
  # shellcheck disable=SC1090
  source $langs_config
else
  # BABEL TRAIN:
  # Amharic - 307
  # Bengali - 103
  # Cantonese - 101
  # Javanese - 402
  # Vietnamese - 107
  # Zulu - 206
  # Dutch (CGN) Fake Babel Code - 505
  # BABEL TEST:
  # Georgian - 404
  # Lao - 203
  # cross lingual setting
  babel_langs="103 107 206 307 402 404 505"
  babel_dev="103 107 206 307 402 404 505"
  babel_recog="101 203 103 107 206 307 402 404 505"
  gp_langs="Czech French Mandarin Thai Bulgarian German Turkish Portuguese"
  gp_dev="Czech French Mandarin Thai Bulgarian German Turkish Portuguese"
  gp_recog="Spanish Polish Croatian Czech French Mandarin Thai Bulgarian German Turkish Portuguese"
  # multilingual setting
  #babel_langs="103 107 206 307 402 404 505 101 203"
  #babel_dev="103 107 206 307 402 404 505 101 203"
  #babel_recog="103 107 206 307 402 404 505 101 203"
  #gp_langs="Czech French Mandarin Thai Bulgarian German Turkish Portuguese Spanish Polish Croatian"
  #gp_dev="Czech French Mandarin Thai Bulgarian German Turkish Portuguese Spanish Polish Croatian"
  #gp_recog="Czech French Mandarin Thai Bulgarian German Turkish Portuguese Spanish Polish Croatian"
  mboshi_train=false
  mboshi_recog=false
  gp_romanized=false
  ipa_transcript=true
fi

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Train Directories
train_set=train
train_dev=dev

# LM Directories
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
lm_train_set=data/local/train.txt
lm_valid_set=data/local/dev.txt

recog_set=""
for l in ${babel_recog} ${gp_recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: Setting up individual languages"
  local/setup_languages.sh \
    --exp_tag "${exp_tag}" \
    --langs "${babel_langs}" \
    --dev "${babel_dev}" \
    --recog "${babel_recog}" \
    --gp-langs "${gp_langs}" \
    --gp-dev "${gp_dev}" \
    --gp-recog "${gp_recog}" \
    --mboshi-train "${mboshi_train}" \
    --mboshi-recog "${mboshi_recog}" \
    --gp-romanized "${gp_romanized}" \
    --ipa-transcript "${ipa_transcript}"

  for x in ${train_set} ${train_dev} ${recog_set}; do
	  sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
  done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Feature extraction"
  fbankdir=fbank
  # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
  for x in ${train_set} ${train_dev} ${recog_set}; do
    utils/fix_data_dir.sh data/${x}
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 30 --write_utt2num_frames true \
        data/${x} exp/make_fbank/${x} ${fbankdir}
    utils/fix_data_dir.sh data/${x}
  done

  mv data/${train_set} data/${train_set}_org
  mv data/${train_dev} data/${train_dev}_org
  remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
  remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

  # compute global CMVN
  compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
  utils/fix_data_dir.sh data/${train_set}

  exp_name=$(basename $PWD)
  # dump features for training
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl /export/b{10,11,12,13}/${USER}/espnet-data/egs/babel/${exp_name}/dump/${train_set}/delta${do_delta}/storage ${feat_tr_dir}/storage
      utils/create_split_dir.pl /export/b{10,11,12,13}/${USER}/espnet-data/egs/babel/${exp_name}/dump/${train_dev}/delta${do_delta}/storage ${feat_dt_dir}/storage
  elif [ -n "$(hostname | grep -i ifp-40)" ] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl /ws/ifp-54_2/hasegawa/${USER}/espnet-data/egs/babel/${exp_name}/dump/${train_set}/delta${do_delta}/storage ${feat_tr_dir}/storage
      utils/create_split_dir.pl /ws/ifp-54_2/hasegawa/${USER}/espnet-data/egs/babel/${exp_name}/dump/${train_dev}/delta${do_delta}/storage ${feat_tr_dir}/storage
  elif [ -n "$(hostname | grep -i ifp)" ] && [ ! -d ${feat_tr_dir}/storage ]; then
      utils/create_split_dir.pl /ws/rz-cl-3/hasegawa/${USER}/espnet-data/egs/babel/${exp_name}/dump/${train_set}/delta${do_delta}/storage ${feat_tr_dir}/storage
      utils/create_split_dir.pl /ws/rz-cl-3/hasegawa/${USER}/espnet-data/egs/babel/${exp_name}/dump/${train_dev}/delta${do_delta}/storage ${feat_tr_dir}/storage
  fi
  dump.sh --cmd "$train_cmd" --nj 20 --do_delta ${do_delta} \
      data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
  dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
      data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
  for rtask in ${recog_set}; do
      feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
      dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
  done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    # The grep trick prevents grep from returning non-zero value when no special symbol is found,
    # which would have prematurely ended the script.
    cut -f 2- -d' ' data/${train_set}/text | tr " " "\n" | sort | uniq | { grep "<" || true; } > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | grep -v '<unk>' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # python retain_lang_feat.py --retain-langs=${retain_lang} --feat-path ${feat_tr_dir}/feats.scp
    # python retain_lang_feat.py --retain-langs=${retain_lang} --feat-path ${feat_dt_dir}/feats.scp

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done

    # remove redundant language
    # python retain_lang.py --retain-langs=${retain_lang} --json-path ${feat_tr_dir}/data.json
    # python retain_lang.py --retain-langs=${retain_lang} --json-path ${feat_dt_dir}/data.json
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
