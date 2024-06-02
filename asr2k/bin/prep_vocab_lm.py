from pathlib import Path
import argparse
from asr2k.lm.inventory import prep_inventory_from_tokens
from asr2k.lm.pronunciation import prep_pronunciation_from_vocab
from asr2k.lm.lexicon import prep_lexicon
from asr2k.lm.grammar import prep_grammar_from_vocab
from asr2k.lm.compile import compile_HLG
import torch

def prep_vocab_lm(lang_dir, lang_id, force=False):

    assert (lang_dir / 'vocab.txt').exists()

    print("step 1: prep pronunciation")
    prep_pronunciation_from_vocab(lang_dir, lang_id, force=force)

    print("step 2 prep L graph")
    prep_lexicon(lang_dir)

    print("step 3: prep inventory")
    prep_inventory_from_tokens(lang_dir)

    print("step 4: prep grammar")
    prep_grammar_from_vocab(lang_dir, force=force)

    print("step 5: combine hlg")
    HLG = compile_HLG(lang_dir, 1)
    torch.save(HLG.as_dict(), f"{lang_dir}/HLG1.pt")



if __name__ == '__main__':

    parser = argparse.ArgumentParser('a utility to update phone database')
    parser.add_argument('--lang_dir', help='lang dir')
    parser.add_argument('--lang_id', help='lang id')
    parser.add_argument('--force', type=bool, default=False, help='force rebuilding')

    args = parser.parse_args()

    lang_dir = Path(args.lang_dir)
    lang_id = args.lang_id

    assert (lang_dir / 'vocab.txt').exists()

    prep_vocab_lm(lang_dir, lang_id, force=args.force)