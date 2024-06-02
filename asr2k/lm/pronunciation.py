from pathlib import Path
from transphone import read_tokenizer
import tqdm
from pathlib import Path
import argparse


def prep_pronunciation_from_vocab(lang_dir, lang_id, g2p=None, force=False):
    prep_pronunciation(lang_dir / 'vocab.txt', lang_dir, lang_id, text_format='text', g2p=g2p)


def prep_vocab_from_text(lang_dir, text_format='text'):

    vocab = set()
    for line in open(lang_dir / 'train_raw.txt', 'r'):
        fields = line.strip().lower().split()
        if text_format == 'kaldi':
            raw_words = fields[1:]
        else:
            raw_words = fields

        vocab.update(raw_words)

    w = open(lang_dir / 'vocab.txt', 'w')
    for word in vocab:
        if word.startswith('<'):
            continue
        w.write(word+'\n')
    w.close()

def prep_pronunciation(text_file, lang_dir, lang_id, text_format='kaldi', g2p=None, force=False):

    if not force and (lang_dir / 'lexicon.txt').exists():
        return

    if g2p is None:
        g2p = read_tokenizer(lang_id)

    lang_dir.mkdir(parents=True, exist_ok=True)

    lexicon_w = open(lang_dir / 'lexicon.txt', 'w')
    lexicon_w.write('<SIL>\tSIL\n<UNK>\tSIL\n')

    p2g_dict = {}

    r = open(text_file, 'r').readlines()

    for line in tqdm.tqdm(r):
        fields = line.strip().split()

        if text_format == 'kaldi':
            raw_words = fields[1:]
        else:
            raw_words = fields

        words = []

        for word in raw_words:
            if word in set(('(', ')', '-', "¿", "“", "¡")):
                continue
            if word.isspace():
                continue

            words.append(word.lower())

        for word in words:

            if word not in p2g_dict:
                pron = g2p.tokenize(word)
                p2g_dict[word] = pron

    for word, pro_lst in p2g_dict.items():
        if len(pro_lst) != 0:
            lexicon_w.write(word+'\t'+' '.join(pro_lst)+'\n')

    lexicon_w.close()

