# text-to-sound

Text-to-audio diffusion, built from scratch: type a description ("rain on a tin roof"), get a ~3 s sound.

Pipeline: caption → CLAP text embedding → diffusion model generates a mel spectrogram → BigVGAN vocoder → waveform.

## Notebooks

Each notebook runs on Kaggle and syncs here on **Save Version** (one file per notebook, at the repo root).

| notebook | phase | what it does |
|---|---|---|
| `clotho-explore.ipynb` | 1–2 | Explores Clotho, checks the BigVGAN round trip and CLAP matching, then builds the training data: 25,469 normalized mel crops `[1, 80, 256]` with CLAP audio/text embeddings |
| `text-to-sound-train.ipynb` | 2, 4 | Loads the dataset and checks un-normalize → BigVGAN; later trains the mel-space model |
| `ddpm-mnist.ipynb` | 3 | DDPM from scratch on MNIST: noise schedule, U-Net, training loop, sampler, class conditioning |

## Data

Private Kaggle dataset `clotho-mel-clap`, built by `clotho-explore`:

| file | shape | contents |
|---|---|---|
| `mels.npy` | `[N, 1, 80, 256]` float16 | log-mels normalized to [-1, 1] |
| `clap_audio.npy` | `[N, 512]` | CLAP audio embedding per crop |
| `clap_text.npy` | `[N, 5, 512]` | CLAP embeddings of the crop's 5 captions |
| `mel_stats.npy` | `[2]` | MIN, MAX for undoing the normalization |
| `meta.csv` | N rows | `crop_id, source, split` (split by source clip) |

Row `i` of every file is the same crop.

## Models used

- [BigVGAN v2](https://github.com/NVIDIA/BigVGAN) `nvidia/bigvgan_v2_22khz_80band_256x`: mel → waveform (22,050 Hz, 80 mels, hop 256)
- [CLAP](https://huggingface.co/laion/clap-htsat-unfused) `laion/clap-htsat-unfused`: shared text/audio embeddings
- Data: [Clotho v2.1](https://zenodo.org/records/4783391)
