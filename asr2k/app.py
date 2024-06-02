from asr2k.am.recognizer import read_recognizer
from asr2k.lm.model import read_lm
import torch
import numpy as np

def read_app(lang_id, lang_dir, ngram=None):
    am = read_recognizer(lang_id)
    lm = read_lm(lang_id, lang_dir, ngram)

    return ASR2K(am, lm, lang_id)

class ASR2K:

    def __init__(self, am, lm, lang_id):

        self.am = am
        self.lm = lm
        self.lang_dir = self.lm.lang_dir
        self.lang_id = lang_id

    def predict(self, audio):

        predicted_phones, nnet_output = self.am.recognize(audio, str(self.lang_dir / 'inventory'), logit=True, segment_duration=20)

        print("predicted phones" , predicted_phones)

        sil_frame = nnet_output[:, :1]
        nnet_output = np.concatenate([sil_frame, nnet_output], axis=1)
        nnet_output = torch.unsqueeze(torch.from_numpy(nnet_output), 0)

        predicted_sent = self.lm.decode(nnet_output)
        return predicted_sent