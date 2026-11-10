"""Mean and best-of-n CLAP score over a fixed prompt set: python eval.py [--guidance 4 --steps 25]"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--guidance", type = float)
ap.add_argument("--steps", type = int)
ap.add_argument("--n", type = int, default = 4, help = "takes per prompt")
ap.add_argument("--seed", type = int, default = 0)
args = ap.parse_args()

import app  # noqa: E402

PROMPTS = ["rain falling", "a dog barking", "birds singing", "people talking", "thunder and heavy rain",
           "footsteps on gravel", "a car engine starting", "waves crashing on the shore", "a door slamming",
           "glass breaking", "a crowd cheering", "wind blowing through trees", "a phone ringing", "a cat meowing",
           "applause", "church bell", "keyboard typing", "acoustic guitar", "cough", "siren", "water dripping",
           "a helicopter flying overhead", "laughter", "piano"]

guidance, steps = args.guidance or app.GUIDANCE, args.steps or app.STEPS
mean, best = [], []
for p in PROMPTS:
    _, _, sims = app.takes(p, args.seed, args.n, guidance, steps)
    mean.append(float(sims.mean()))
    best.append(float(sims.max()))
    print(f"{p:32s} mean {mean[-1]:+.3f}  best {best[-1]:+.3f}", flush = True)
print(f"{app.UNET_FILE} · cfg {guidance} · {steps} steps · {args.n} takes: "
      f"mean {sum(mean) / len(mean):.3f}  best {sum(best) / len(best):.3f}")
