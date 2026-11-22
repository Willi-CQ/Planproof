"""Optional text-only diagnostic. Load the trained linear classifier without pickle or scikit-learn."""
from collections import Counter
from functools import lru_cache
import hashlib
import re
import unicodedata

import numpy as np


DIMS = 4096
ROOM = re.compile(r'\b(?:R[.-]?\d{1,5}|(?:BED|BATH)\d+|CORRIDOR|HALLWAY|KITCHEN|FLUR|TREPPENRAUM)\b',re.I)
DOOR = re.compile(r'\b(?:DOOR|DR|D)[.-]?\d{1,5}[A-Z]?\b',re.I)
EXCLUDED = re.compile(r'\b(?:missing|except|without|minimum|regulations?|compliant|should|must|'
                      r'calculate|calculat(?:ed|ing)|scale|from the dimensions|date|architect|egress|'
                      r'außer|ohne|fehlend\w*|muss|müssen|berechn\w*|maßstab|datum|architekt\w*|'
                      r'first floor|ground floor|upper floor|erdgeschoss|obergeschoss|every floor|'
                      r'inches|feet|centimetres|centimeters|zentimeter\w*|cm)\b',re.I)


@lru_cache(maxsize=8192)
def bucket(feature):
    return int.from_bytes(hashlib.blake2s(feature.encode(),digest_size=4).digest(),'little') % DIMS


def features(text):
    text = unicodedata.normalize('NFKC',text).casefold()
    text = DOOR.sub('doortoken',ROOM.sub('roomtoken',text))
    words = re.findall(r'\w+',text)
    tokens = ['w:'+word for word in words] + ['b:'+a+' '+b for a,b in zip(words,words[1:])]
    padded = ' '+ ' '.join(words)+' '
    for n in (3,4,5):
        tokens.extend(padded[i:i+n] for i in range(len(padded)-n+1))
    vector = np.zeros(DIMS,dtype=np.float32)
    for k,count in Counter(bucket(t) for t in tokens).items():
        vector[k] = 1 + np.log(count)
    return vector / max(1., np.linalg.norm(vector))


class IntentReader:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as model:
            self.weights = model['weights']
            self.intercept = model['intercept']
            self.labels = model['labels']

    def predict(self, question):
        unknown = dict(op='unsupported',args={})
        values = features(question) @ self.weights.T + self.intercept
        first, second = np.argsort(values)[-2:][::-1]
        # Fixed acceptance margins are defined before opening the independently written questions.
        if values[first] < -.25 or values[first]-values[second] < .13 or EXCLUDED.search(question):
            return unknown
        op = str(self.labels[first])
        rooms = [m.upper() for m in ROOM.findall(question)]
        doors = [m.upper() for m in DOOR.findall(question)]
        if op in ('count_marks','project_title') and not rooms and not doors:
            return dict(op=op,args={})
        if op == 'adjacent_doors' and len(rooms)==2 and len(set(rooms))==2 and not doors:
            return dict(op=op,args={'rooms':rooms})
        if op in ('room_door_count','printed_area') and len(rooms)==1 and not doors:
            return dict(op=op,args={'room':rooms[0]})
        if op in ('as_built_width','fire_rating','locate_mark') and len(doors)==1 and not rooms:
            p = dict(op=op,args={'mark':doors[0]})
            if op=='fire_rating' and re.search(r'\b(?:certified|zertifiziert\w*)\b',question,re.I):
                p['certified_only'] = True
            return p
        return unknown
