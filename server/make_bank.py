"""Builds clap_audio_bank.npy from the training clips: python make_bank.py <data folder>"""

import glob
import os
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1]
train = (pd.read_csv(os.path.join(DATA, "meta.csv")).split == "train").values
bank = [np.load(os.path.join(DATA, "clap_audio.npy"))[train]]
for f in sorted(glob.glob(os.path.join(DATA, "fsd", "shard_*.npz"))):
    bank.append(np.load(f)["clap_audio"])
print("clotho", len(bank[0]), "| fsd", sum(len(b) for b in bank[1:]))
np.save("clap_audio_bank.npy", np.concatenate(bank).astype(np.float16))
