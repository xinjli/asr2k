from pathlib import Path
import argparse
from asr2k.lm.inventory import prep_inventory_from_tokens
from asr2k.lm.pronunciation import prep_pronunciation_from_vocab
from asr2k.lm.lexicon import prep_lexicon
from asr2k.lm.grammar import prep_grammar_from_ngram
from asr2k.lm.compile import compile_HLG
import torch

def prep_ngram_lm(lang_dir, lang_id, ngram=2, force=False):

    if ngram == 2:
        assert (lang_dir / f'bigrams.txt').exists()
    elif ngram == 1:
        assert (lang_dir / f'unigrams.txt').exists()
    elif ngram == 3:
        assert (lang_dir / f'trigrams.txt').exists()


    print("step 1: prep pronunciation")
    prep_pronunciation_from_vocab(lang_dir, lang_id, force=force)

    print("step 2 prep L graph")
    prep_lexicon(lang_dir)

    print("step 3: prep inventory")
    prep_inventory_from_tokens(lang_dir)

    print("step 4: prep grammar")
    prep_grammar_from_ngram(lang_dir, ngram=ngram, force=force)

    print("step 5: combine hlg")
    HLG = compile_HLG(lang_dir, ngram)
    torch.save(HLG.as_dict(), f"{lang_dir}/HLG{ngram}.pt")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('a utility to update phone database')
    parser.add_argument('--lang_dir', help='lang dir')
    parser.add_argument('--lang_id', help='lang id')
    parser.add_argument('--ngram', default=2, type=int, help='ngram')
    parser.add_argument('--force', type=bool, default=False, help='force rebuilding')

    args = parser.parse_args()

    lang_dir = Path(args.lang_dir)
    lang_id = args.lang_id
    ngram = args.ngram

    assert (lang_dir / 'vocab.txt').exists()

    prep_ngram_lm(lang_dir, lang_id, ngram, force=args.force)