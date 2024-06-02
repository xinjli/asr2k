import logging
from pathlib import Path
import k2
import torch
from icefall.lexicon import Lexicon

#
# def prep_hlg(text_path, lang_id, lang_dir, train_length=-1, force=False, lexicon_path=None):
#
#     text_path = Path(text_path)
#     lang_dir = Path(lang_dir)
#
#     if not lang_dir.exists():
#         lang_dir.mkdir(parents=True, exist_ok=True)
#
#     split_text(text_path, lang_dir, 100, train_length)
#
#     if lexicon_path is None:
#         prep_lexicon(lang_id, lang_dir, force)
#     else:
#         print("copying lexicon from {lexicon_path} to {lang_dir}")
#         shutil.copyfile(lexicon_path, lang_dir / 'lexicon.txt')
#
#     prep_lang(lang_dir)
#     prep_inventory(lang_dir)
#     prep_grammar(text_path, lang_dir, force)
#
#     HLG2 = compile_HLG(lang_dir, 2)
#     print(f"Saving HLG2.pt to {lang_dir}")
#     torch.save(HLG2.as_dict(), f"{lang_dir}/HLG2.pt")
#
#     HLG3 = compile_HLG(lang_dir, 3)
#     print(f"Saving HLG3.pt to {lang_dir}")
#     torch.save(HLG3.as_dict(), f"{lang_dir}/HLG3.pt")
#
#
# def prep_crubadan_hlg(lang_id, lang_dir, force=False):
#
#     lang_dir = Path(lang_dir)
#
#     if not lang_dir.exists():
#         lang_dir.mkdir(parents=True, exist_ok=True)
#
#     #crubadan_path = Path('/home/xinjianl/Git/asr2k/data/crubadan')
#     #crubadan_lang_path = crubadan_path / lang_id
#
#     if not lang_dir.exists():
#         print(f'lang id {lang_id} is not available in crubadan')
#         return False
#
#     #prep_crubadan_lexicon(crubadan_lang_path, lang_id, lang_dir, force)
#     prep_lang(lang_dir)
#     prep_inventory(lang_dir)
#     prep_crubadan_grammar(lang_dir, force)
#
#     HLG1 = compile_HLG(lang_dir, 1)
#     print(f"Saving HLG1.pt to {lang_dir}")
#     torch.save(HLG1.as_dict(), f"{lang_dir}/HLG1.pt")
#
#     HLG2 = compile_HLG(lang_dir, 2)
#     print(f"Saving HLG2.pt to {lang_dir}")
#     torch.save(HLG2.as_dict(), f"{lang_dir}/HLG2.pt")



def compile_HLG(lang_dir, ngram) -> k2.Fsa:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang_phone or data/lang_bpe_5000.

    Return:
      An FSA representing HLG.
    """
    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)
    logging.info(f"Building ctc_topo. max_token_id: {max_token_id}")
    H = k2.ctc_topo(max_token_id)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))

    if Path(f"{lang_dir}/G_{ngram}_gram.pt").is_file():
        logging.info(f"Loading pre-compiled G_{ngram}_gram")
        d = torch.load(f"{lang_dir}/G_{ngram}_gram.pt")
        G = k2.Fsa.from_dict(d)
    else:
        logging.info(f"Loading G_{ngram}_gram.fst.txt")
        with open(lang_dir / f"G_{ngram}_gram.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            torch.save(G.as_dict(), f"{lang_dir}/G_{ngram}_gram.pt")

    first_token_disambig_id = lexicon.token_table["#0"]
    first_word_disambig_id = lexicon.word_table["#0"]

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(f"LG shape after k2.connect: {LG.shape}")

    logging.info(type(LG.aux_labels))
    logging.info("Determinizing LG")

    LG = k2.determinize(LG)
    logging.info(type(LG.aux_labels))

    logging.info("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    logging.info("Removing disambiguation symbols on LG")

    LG.labels[LG.labels >= first_token_disambig_id] = 0
    # See https://github.com/k2-fsa/k2/issues/874
    # for why we need to set LG.properties to None
    LG.__dict__["_properties"] = None

    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing H and LG")
    # CAUTION: The name of the inner_labels is fixed
    # to `tokens`. If you want to change it, please
    # also change other places in icefall that are using
    # it.
    HLG = k2.compose(H, LG, inner_labels="tokens")

    logging.info("Connecting LG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)
    logging.info(f"HLG.shape: {HLG.shape}")

    return HLG
