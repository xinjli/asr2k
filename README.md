# ASR2K

This repository contains our code of our publication at `interspeech 2022`

```
Li, Xinjian, et al. "ASR2K: Speech Recognition for Around 2000 Languages without Audio" Interspeech 2022. 2022
```

We plan to release ASR models for 2k languages (currently 1909 languages). The architecture is as follows:

![asr2k](./arch.png)

## Usage

### Training

See README in `egs/commonvoice` for a simple recipe example

### Inference

Once you trained a model or download one of our pretrained model (not available yet). 

You should be able to run it using python as follows

```python
In [1]: from asr2k.app import read_app
In [2]: app = read_app('eng', './data')
In [3]: app.predict('utt.wav')
```

or run inference from bash

```bash
python -m asr2k.bin.run --lang=eng --lang_dir=./data --input=./test --output=./test
```


## Install

To train a ASR2K model, you need the following packages:

```bash
# k2
# https://k2-fsa.github.io/k2/
# in my env, it is the following
pip install k2==1.24.4.dev20240223+cpu.torch1.13.1 -f https://k2-fsa.github.io/k2/cpu.html

# lhotse
pip install git+https://github.com/lhotse-speech/lhotse

# icefall
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt


# srilm
# download from http://www.speech.sri.com/projects/srilm/download.html
# follow the instruction in the INSTALL file in the package
# in my env, they are
# 
# - tar -xvzf srilm-1.7.3.tar.gz
# - set SRILM variable in Makefile
# - make
# - add bin/i686-m64 to your PATH
# 
# make sure ngram-count is available in your env

# ASR2k
# install this package
pip install -e .
```


## Reference

```
@inproceedings{li22aa_interspeech,
  author={Xinjian Li and Florian Metze and David R. Mortensen and Alan W Black and Shinji Watanabe},
  title={{ASR2K: Speech Recognition for Around 2000 Languages without Audio}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4885--4889},
  doi={10.21437/Interspeech.2022-10712}
}
```