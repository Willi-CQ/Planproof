import argparse
import json
from pathlib import Path

from . import answer,build_scene


def main():
    p=argparse.ArgumentParser(description='Answer a question against one original plan page and return evidence.')
    p.add_argument('drawing',type=Path)
    p.add_argument('question',nargs='+')
    p.add_argument('--page',type=int,default=1)
    p.add_argument('--arm',choices=('facts','relations'),default='relations')
    a=p.parse_args()
    scene=build_scene(a.drawing,a.page)
    print(json.dumps(answer(scene,' '.join(a.question),a.arm),ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
