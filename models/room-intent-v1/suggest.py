"""Print a proposed room-reader operation for a German or English question."""
import argparse
import hashlib
import json
from pathlib import Path

from intent import IntentReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='+')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model = root/'bilingual-intent.npz'
    receipt = json.loads((root/'fit-receipt.json').read_bytes())
    if hashlib.sha256(model.read_bytes()).hexdigest() != receipt['model_sha256']:
        raise ValueError('Model hash does not match the released fit.')
    question = ' '.join(args.question)
    suggestion = IntentReader(model).predict(question)
    print(json.dumps(dict(mode='suggestion', model_sha256=receipt['model_sha256'],
                          program=suggestion), ensure_ascii=False))


if __name__ == '__main__':
    main()
