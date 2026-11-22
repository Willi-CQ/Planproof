"""Train the linear intent model on the supplied bilingual phrases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.svm import LinearSVC

from intent import features


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New output directory for the fitted weights and receipt')
    out = parser.parse_args().output
    out.mkdir(parents=True)
    raw = (root/'train.tsv').read_bytes()
    rows = list(csv.reader(raw.decode('utf8').splitlines(),delimiter='\t'))
    ids = dict(a='R73',b='R84',d='D92')
    x = np.array([features(q.format(**ids)) for _,q in rows])
    y = [label for label,_ in rows]
    start = time.process_time()
    fit = LinearSVC(C=2,random_state=724,dual=True,max_iter=5000).fit(x,y)
    seconds = time.process_time()-start
    model = out/'bilingual-intent.npz'
    np.savez_compressed(model,weights=fit.coef_.astype(np.float32),
                        intercept=fit.intercept_.astype(np.float32),labels=fit.classes_)
    (out/'receipt.json').write_text(json.dumps(dict(training_sha256=hashlib.sha256(raw).hexdigest(),
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),code_sha256=hashlib.sha256((root/'intent.py').read_bytes()).hexdigest(),
        training_phrasings=len(rows),classes=list(fit.classes_),cpu_seconds=seconds,bytes=model.stat().st_size,paid_eur=0),indent=2))


if __name__ == '__main__':
    main()
