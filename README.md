# Zero-Shot Cross-Lingual Phonetic Recognition with External Language Embedding
This repository is an implementation for [Zero-Shot Cross-Lingual Phonetic Recognition with External Language Embedding](https://www.isca-speech.org/archive/interspeech_2021/gao21_interspeech.html). It is a recipe of ESPnet.

* Please follow the instruction in setup.sh
  - clone the specific version of ESPnet and install it.
  - copy the additional model, training and evaluation scripts to ESPnet.
* Run prepare.sh to setup the multilingual corpora.
* Run run.sh to train and evaluate the model.

If you find this project useful, please consider citing this work.
``` bibtex
@inproceedings{gao2021zero,
  title={Zero-shot cross-lingual phonetic recognition with external language embedding},
  author={Gao, Heting and Ni, Junrui and Zhang, Yang and Qian, Kaizhi and Chang, Shiyu and Hasegawa-Johnson, Mark},
  booktitle={22nd Annual Conference of the International Speech Communication Association, INTERSPEECH 2021},
  pages={4426--4430},
  year={2021},
  organization={International Speech Communication Association}
}
```
