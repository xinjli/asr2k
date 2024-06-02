# README

This is a simple recipe to build a ASR2K language model from Common Voice English using 1k lines of text only

## Train

first download CV english dataset from Mozilla website and then see train.sh for details.

It will generate a `data` directory which contains all language model

## Inference

Use the trained model and run inference over 1 utterance (stored in test directory).

It will generate `decode.txt` in `test` directory

```bash
python -m asr2k.bin.run --lang=eng --lang_dir=./data --input=./test --output=./test
```

## eval

You can compare ref and hyp as follows.

```
python -m asr2k.bin.eval --ref=./test/ref.txt --hyp=./test/decode.txt
cer:  0.21052631578947367
wer:  0.4
```