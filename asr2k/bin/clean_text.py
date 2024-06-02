from asr2k.lm.cleaner import Cleaner
from pathlib import Path
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('run inference')
    parser.add_argument('-i', '--input', required=True, help='input file')
    parser.add_argument('-o', '--output', default='stdout', help='output directory')
    parser.add_argument('-l', '--lang', default='eng', help='lang id')
    parser.add_argument('--force', type=bool, default=False, help='force rebuilding')
    
    cleaner = Cleaner()
    
    args = parser.parse_args()
    input_file = args.input
    
    w = open(args.output, 'w')
    
    for line in open(input_file, 'r'):
        cleaned_line = cleaner.clean(line.strip())
        w.write(cleaned_line+'\n')

    w.close()