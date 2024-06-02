from pathlib import Path
import argparse
from asr2k.app import read_app


def run(input, output, lang, lang_dir):

    app = read_app(lang, lang_dir)

    w = None

    if output != 'stdout':
        Path(output).mkdir(parents=True, exist_ok=True)
        w = open(Path(output) / 'decode.txt', 'w')

    if input.is_dir():
        for f in sorted(input.iterdir()):
            
            if not str(f).endswith(".wav"):
                continue
            
            print(f"processing {f}")
            utt_id = f.stem
            res = app.predict(f)

            if w is not None:
                w.write(f"{utt_id} {res}\n")
            else:
                print(f"{utt_id} {res}")

        w.close()
    else:
        print(app.predict(args.input))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('run inference')
    parser.add_argument('-i', '--input', required=True, help='input file')
    parser.add_argument('-o', '--output', default='stdout', help='output directory')
    parser.add_argument('-l', '--lang', default='eng', help='lang id')
    parser.add_argument('--lang_dir', help='lang dir')
    parser.add_argument('--force', type=bool, default=False, help='force rebuilding')

    args = parser.parse_args()
    input = Path(args.input)
    output = Path(args.output)
    lang = args.lang
    lang_dir = args.lang_dir

    run(input, output, lang, lang_dir)