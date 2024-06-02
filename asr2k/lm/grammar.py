import os
import k2
import torch

def prep_grammar(text_path, lang_dir, force=False):

    if not force and (lang_dir / "G_2_gram.pt").exists() and (lang_dir / "G_3_gram.pt").exists():
        print("Grammar already built. skipping")
        return

    lang_dir = lang_dir.absolute()

    os.system(f"lmplz -o 3 < {text_path} > {lang_dir/'text.arpa'}")

    os.system(f"python3 -m kaldilm --read-symbol-table='{lang_dir}/words.txt' --disambig-symbol='#0' --max-order=2 {lang_dir}/text.arpa > {lang_dir}/G_2_gram.fst.txt")
    os.system(f"python3 -m kaldilm --read-symbol-table='{lang_dir}/words.txt' --disambig-symbol='#0' --max-order=3 {lang_dir}/text.arpa > {lang_dir}/G_3_gram.fst.txt")

    with open(lang_dir / f"G_2_gram.fst.txt") as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        torch.save(G.as_dict(), f"{lang_dir}/G_2_gram.pt")

    with open(lang_dir / f"G_3_gram.fst.txt") as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        torch.save(G.as_dict(), f"{lang_dir}/G_3_gram.pt")


def prep_ngram_from_text(lang_dir, ngram=2, force=False):

    output_file = "unigrams.txt"
    if ngram == 2:
        output_file = "bigrams.txt"
    elif ngram == 3:
        output_file = "trigrams.txt"

    os.system(f"ngram-count -text {lang_dir / 'train_raw.txt'} -order {ngram} -tolower -write {lang_dir / output_file}")


def prep_grammar_from_ngram(lang_dir, ngram=2, force=False):

    if not force and (lang_dir / f"G_{ngram}_gram.pt").exists():
        print("Grammar already built. skipping")
        return

    lang_dir = lang_dir.absolute()
    #normalized_bigram = open(lang_dir / 'bigrams.txt', 'w')

    #for line in open(lang_dir, 'r'):
    #    normalized_bigram.write(line.lower())

    #normalized_bigram.close()

    if ngram == 3:
        os.system(f"ngram-count -order 3 -read {lang_dir / 'trigrams.txt'} -lm {lang_dir / 'text3.arpa'}")
        os.system(f"python3 -m kaldilm --read-symbol-table='{lang_dir}/words.txt' --disambig-symbol='#0' --max-order=3 {lang_dir}/text3.arpa > {lang_dir}/G_3_gram.fst.txt")
        with open(lang_dir / f"G_3_gram.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            torch.save(G.as_dict(), f"{lang_dir}/G_3_gram.pt")
    elif ngram == 2:
        os.system(f"ngram-count -order 2 -read {lang_dir / 'bigrams.txt'} -lm {lang_dir / 'text2.arpa'}")
        os.system(f"python3 -m kaldilm --read-symbol-table='{lang_dir}/words.txt' --disambig-symbol='#0' --max-order=2 {lang_dir}/text2.arpa > {lang_dir}/G_2_gram.fst.txt")
        with open(lang_dir / f"G_2_gram.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            torch.save(G.as_dict(), f"{lang_dir}/G_2_gram.pt")
    elif ngram == 1:
        os.system(f"ngram-count -order 1 -read {lang_dir / 'unigrams.txt'} -lm {lang_dir / 'text1.arpa'}")

        os.system(f"python3 -m kaldilm --read-symbol-table='{lang_dir}/words.txt' --disambig-symbol='#0' --max-order=1 {lang_dir}/text1.arpa > {lang_dir}/G_1_gram.fst.txt")
        with open(lang_dir / f"G_1_gram.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            torch.save(G.as_dict(), f"{lang_dir}/G_1_gram.pt")


def prep_grammar_from_vocab(lang_dir, force=False):

    assert (lang_dir / 'vocab.txt').exists(), f"vocab.txt not found in {lang_dir}"
    assert (lang_dir / 'words.txt').exists(), f"words.txt not found in {lang_dir}"

    if not force and (lang_dir / f"G_1_gram.pt").exists():
        print("Grammar already built. skipping")
        return

    # build unigram file
    if not (lang_dir / 'unigrams.txt').exists():
        w = open(lang_dir / 'unigrams.txt', 'w')

        for line in open(lang_dir / 'vocab.txt', 'r'):
            w.write(line.strip()+' ' + str(1) + '\n')

        w.close()

    os.system(f"ngram-count -order 1 -read {lang_dir / 'unigrams.txt'} -lm {lang_dir / 'text1.arpa'}")
    os.system(f"python3 -m kaldilm --read-symbol-table='{lang_dir}/words.txt' --disambig-symbol='#0' --max-order=1 {lang_dir}/text1.arpa > {lang_dir}/G_1_gram.fst.txt")
    with open(lang_dir / f"G_1_gram.fst.txt") as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        torch.save(G.as_dict(), f"{lang_dir}/G_1_gram.pt")