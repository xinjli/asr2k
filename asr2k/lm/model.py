from asr2k.config import read_config
from itertools import groupby
import editdistance
from collections import defaultdict
from asr2k.am.module.ssl_transformer import SSLTransformer
from asr2k.config import lang_path
from sys import exit
import numpy as np
from asr2k.utils.tensor import tensor_to_cuda, move_to_tensor, move_to_ndarray
from asr2k.bin.download_model import download_language_model
import k2
from icefall.decode import get_lattice, one_best_decoding
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)
import torch
from pathlib import Path


def read_lm(lang_id, lang_dir, ngram=None):
    lm = LanguageModel(Path(lang_dir), lang_id, ngram)
    return lm


class LanguageModel:

    def __init__(self, lang_dir, lang_id, ngram=None):

        self.lang_dir = lang_dir
        self.lang_id = lang_id
        self.ngram = ngram

        if self.ngram is None:
            hlg_file = 'HLG.pt'
            for i in range(3, 0, -1):
                if (lang_dir / f'HLG{i}.pt').exists():
                    hlg_file = f'HLG{i}.pt'
                    break
        else:
            hlg_file = f'HLG{ngram}.pt'

        assert lang_dir / hlg_file, f'HLG{ngram}.pt not found in {lang_dir}'

        self.HLG = k2.Fsa.from_dict(
            torch.load(lang_dir / hlg_file)
        )

        self.lexicon = Lexicon(lang_dir)

        self.params = {
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }

    def decode(self, nnet_output):

        batch_size = nnet_output.shape[0]
        supervision_segments = torch.tensor(
            [[i, 0, nnet_output.shape[1]] for i in range(batch_size)],
            dtype=torch.int32,
        )

        lattice = get_lattice(
            nnet_output=nnet_output,
            decoding_graph=self.HLG,
            supervision_segments=supervision_segments,
            search_beam=self.params["search_beam"],
            output_beam=self.params["output_beam"],
            min_active_states=self.params["min_active_states"],
            max_active_states=self.params["max_active_states"],
        )

        best_path = one_best_decoding(
            lattice=lattice, use_double_scores=self.params["use_double_scores"]
        )
        hyps = get_texts(best_path)
        hyps = [[self.lexicon.word_table[i] for i in ids] for ids in hyps]
        predicted = ' '.join(hyps[0])

        return predicted