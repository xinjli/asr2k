# change DB_PATH to yours
DB_PATH=/ocean/projects/cis210027p/shared/corpora/commonvoice/cv-corpus-11.0-2022-09-21/cv-corpus-11.0-2022-09-21

if [ ! -d "$DB_PATH" ]; then
  echo "Error: Directory $DB_PATH does not exist. change it to your local CV path"
  exit 1
fi

echo "prep raw text"
mkdir -p data
head -n 1001 $DB_PATH/en/train.tsv | cut -d $'\t' -f3 | sed '1d' > ./data/raw.txt
python -m asr2k.bin.clean_text -i ./data/raw.txt -o ./data/train_raw.txt

echo "train language model"
python -m asr2k.bin.prep_text_lm --lang_dir=./data --lang_id=eng --ngram=3