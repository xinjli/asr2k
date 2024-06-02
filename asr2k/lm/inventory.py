from pathlib import Path

def prep_inventory_from_tokens(lang_dir):

    inv_dir = lang_dir / 'inventory'
    inv_dir.mkdir(parents=True, exist_ok=True)

    r = open(lang_dir / 'tokens.txt').readlines()[2:]

    phoneme_lst = []

    for line in r:
        if line.startswith('#'):
            break

        phoneme_lst.append(line.strip().split()[0])

    w = open(inv_dir / 'phone.txt', 'w')
    w.write('\n'.join(phoneme_lst))
    w.close()

    w = open(inv_dir / 'phoneme.txt', 'w')
    w.write('\n'.join(phoneme_lst))
    w.close()

    w = open(inv_dir / 'allophone.txt', 'w')
    for p in phoneme_lst:
        w.write(p+' '+p+'\n')
    w.close()
