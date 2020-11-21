from argparse import ArgumentParser
from pathlib import Path

import requests

URL = "http://localhost:5000/api/infer"


def do_post(pth):
    img_path = Path(pth)
    r = requests.post(URL, files={
        'image': open(img_path, 'rb')
    })
    print(r.json())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    do_post(args.path)
