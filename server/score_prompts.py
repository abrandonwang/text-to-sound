"""Rescores good_prompts.json (best seed + CLAP score per prompt): python score_prompts.py --fresh"""

import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type = int, default = 2, help = "seeds tried per prompt")
ap.add_argument("--fresh", action = "store_true", help = "rescore prompts that already have a score")
ap.add_argument("--save", help = "folder to write each prompt's kept take to, for listening")
args = ap.parse_args()

import app  # noqa: E402

data = json.load(open(app.PROMPTS_FILE))
if args.save:
    os.makedirs(args.save, exist_ok = True)

for k, g in enumerate(data["prompts"]):
    if "clap" in g and not args.fresh:
        continue
    best = None
    for seed in range(args.seeds):
        wavs, _, sims = app.takes(g["prompt"], seed)
        i = int(sims.argmax())
        if best is None or float(sims[i]) > best[0]:
            best = (float(sims[i]), seed, wavs[i])
    g["seed"], g["clap"] = best[1], round(best[0], 3)
    print(f"{k + 1}/{len(data['prompts'])} {g['prompt']:32s} seed {g['seed']} clap {g['clap']:.3f}", flush = True)
    if args.save:
        with open(os.path.join(args.save, g["prompt"].replace(" ", "_") + ".wav"), "wb") as f:
            f.write(app.wav_bytes(app.polish(best[2])))
    data["note"] = f"seed and clap from score_prompts.py: {app.UNET_FILE}, cfg {app.GUIDANCE}, {app.STEPS} steps, best of {app.N_TAKES}"
    json.dump(data, open(app.PROMPTS_FILE, "w"), indent = 1)

scores = sorted(g["clap"] for g in data["prompts"])
print(f"median {scores[len(scores) // 2]:.3f} | below {app.MIN_CLAP}: {sum(s < app.MIN_CLAP for s in scores)} of {len(scores)}")
