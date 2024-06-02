import argparse
import editdistance

def eval_error(ref_file, hyp_file):

    utt2ref = {}
    utt2hyp = {}

    for line in open(ref_file):
        fields = line.strip().split()
        utt2ref[fields[0]] = fields[1:]

    for line in open(hyp_file):
        fields = line.strip().split()
        utt2hyp[fields[0]] = fields[1:]

    char_cnt = 0
    word_cnt = 0
    char_err = 0
    word_err = 0

    for utt in utt2hyp:

        if utt not in utt2ref:
            print("utt not in ref: ", utt)
            continue

        hyp_word = utt2hyp[utt]
        ref_word = utt2ref[utt]

        ref_sent = ''.join(ref_word)
        hyp_sent = ''.join(hyp_word)
        char_err += editdistance.eval(ref_sent, hyp_sent)
        char_cnt += len(ref_sent)

        word_err += editdistance.eval(ref_word, hyp_word)
        word_cnt += len(ref_word)

    print("cer: ", char_err/char_cnt)
    print("wer: ", word_err/word_cnt)

    return char_err/char_cnt, word_err/word_cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser('a utility to update phone database')
    parser.add_argument('--ref', type=str, help='ref file')
    parser.add_argument('--hyp', type=str, help='hyp file')
    args = parser.parse_args()
    eval_error(args.ref, args.hyp)


