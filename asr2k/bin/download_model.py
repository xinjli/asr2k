import tarfile
from urllib.request import urlopen
import io
import argparse
from asr2k.config import model_path


def download_acoustic_model(model_name=None, alt_model_path=None):

    if model_name is None:
        model_name = 'latest'
    if alt_model_path:
        model_dir = alt_model_path
    else:
        model_dir = model_path

    if not (model_dir / model_name).exists():

        try:
            url = 'https://github.com/xinjli/allosaurus/releases/download/v1.0/' + model_name + '.tar.gz'
            print("downloading model ", model_name)
            print("from: ", url)
            print("to:   ", str(model_dir))
            print("please wait...")
            resp = urlopen(url)
            compressed_files = io.BytesIO(resp.read())
            files = tarfile.open(fileobj=compressed_files)
            files.extractall(str(model_dir))

        except Exception as e:
            print("Error: could not download the model", e)
            (model_dir / model_name).rmdir()


def download_language_model(model_name=None, alt_model_path=None):
    raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser('a utility to download pretrained models')
    parser.add_argument('-m', '--model', default='latest',  help='specify which model to download. A list of downloadable models are available on Github')

    args = parser.parse_args()

    download_model(args.model)
