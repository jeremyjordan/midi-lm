# MIDI Language Model

*Generative modeling of MIDI files*

When it comes to modeling music, there's two general approaches:

- modeling the raw audio waveforms ([AudioLM][1], [MusicGen][2])
- modeling the symbolic representations ([Music Transformer][3], [Multitrack Music Transformer][4], [SymphonyNet][5],
  [MusicVAE][6])

I'm mostly interested in modeling symbolic representations because it **preserves editability** post-generation. As a
composer, you could go back and make small edits (e.g. shift the pitch of an individual note), repeat certain sections
(e.g. copy and paste a set of notes), or easily modify the instrument for a given track (e.g. turn the piano track into
a trumpet track).

Another useful aspect of symbolic representations is that they're very data-efficient. For example, if we looked at the
[MAESTRO dataset][7], the raw audio representation would take 120GB of storage whereas the symbolic representation
(MIDI) only takes 81MB of storage (perhaps the better comparison would be the number of tokens needed to represent each
type of sequence, but this can vary depending on your tokenization strategy).

[1]: https://arxiv.org/abs/2209.03143
[2]: https://arxiv.org/abs/2306.05284
[3]: https://arxiv.org/abs/1809.04281v2
[4]: https://arxiv.org/abs/2207.06983
[5]: https://arxiv.org/abs/2205.05448
[6]: https://arxiv.org/abs/1803.05428
[7]: https://magenta.tensorflow.org/datasets/maestro

## Project structure

This project is set up to encourage easy experimentation by providing a consistent training structure alongside
interfaces for each of the axes which I might wish to experiment. The main interfaces are defined as:

- **dataset**: a PyTorch Lightning data module which encapsulates training/validation datasets and dataloaders
- **network**: the underlying `torch.nn.Module` architecture
- **model**: a PyTorch Lightning module which defines training and validation steps for a given network
- **tokenizer**: a object which can convert `muspy.Music` objects into a dictionary of tensors and vice versa
- **transforms**: a set of function which can modify a `muspy.Music` object to do things like cropping and transposing

These various objects are assembled in `midi_lm/train.py` according to the configuration passed in by `hydra` from a
command line invocation.

## Supported datasets

I've added multiple different MIDI datasets of varying complexity (from basic scales to full orchestral pieces) along
with networks of varying capacity so I can do some exploration of ideas at a small scale before ramping up the compute
cost to train larger models on bigger datasets.

| Dataset      | Description                                                         | Train (file count)  | Validation (file count)  |
|--------------|---------------------------------------------------------------------|---------------------|--------------------------|
| Eighth Notes | One measure of eighth notes for 12 different pitches                | 6                   | 6                        |
| Scales       | 12 major and 12 minor scales                                        | 20                  | 4                        |
| JSB Chorales | A collection of 382 four-part chorales by Johann Sebastian Bach     | 229                 | 76                       |
| NES          | Songs from the soundtracks of 397 NES games                         | 4441*               | 395*                     |
| MAESTRO      | MIDI recordings from ten years of International Piano-e-Competition | 962                 | 137                      |
| SymphonyNet  | A collection of classical and contemporary symphonic compositions   | 37088               | 9272                     |

*Count after filtering out files from the original dataset which don't meet data quality thresholds for a minimum number
of beats, tracks, etc.

You can see more information about these datasets in [this Weights and Biases
report](https://api.wandb.ai/links/jeremytjordan/jtwn1s8s).

## Supported model architectures

I aim to support a mixture of reference implementations from the literature alongside various ideas that I'm exploring.

- [**Multitrack Music Transformer**](https://salu133445.github.io/mmt/) ([paper][mmt1], [code][mmt2]): a standard
  Transformer architecture which represents MIDI files as a sequence of (event_type, beat, position, pitch, duration,
  instrument) tokens. This model has 6 input/output heads, one for each different token type.
- **Multi-head transformer**: a slightly more generic version architecture which supports arbitrary input/output heads
  which are merged and passed into a decoder-only Transformer model. This architecture can be used for other
  tokenization strategies such as the (time-shift, pitch, duration) tokenizer for single-track MIDI files.
- **Structured transformer**: a standard decoder-only Transformer architecture (only one input/output head) which
  enforces an explicit sampling structure during inference, this model expects tokens to appear in repeating sets of
  (pitch, velocity, duration, time_shift) tokens.

[mmt1]: https://arxiv.org/abs/2207.06983
[mmt2]: https://github.com/salu133445/mmt

## Local setup

Initialize the environment
```
make bootstrap
```

Install the requirements
```
make install
```

## Training a model

Model training can be kicked off from the command line using [Hydra](https://hydra.cc/). Hydra provides a lot of control
over how you compose configurations from the command line. These configurations are defined in `midi_lm/config/` as
dataclass objects. These dataclass objects are given short-names in the "config store" defined in
`midi_lm/config/__init__.py`.

Example local runs:
```
train compute=local logger=wandb-test trainer=mps \
    tokenizer=mmt model=mmt network=mmt-small \
    dataset=scales transforms=crop \
    trainer.val_check_interval=1.0 trainer.max_epochs=20
```
```
train compute=local logger=wandb-test trainer=mps \
    tokenizer=mmt model=mmt network=mmt-small \
    dataset=bach transforms=crop_transpose \
    trainer.val_check_interval=1.0
```
```
train compute=local logger=wandb-test trainer=mps \
    tokenizer=mmt model=mmt network=mmt-small \
    dataset=nes transforms=crop-transpose
```

Example remote run:
```
train compute=a10g compute.timeout=14400 logger=wandb-test trainer=gpu \
    tokenizer=mmt model=mmt network=mmt \
    dataset=maestro transforms=crop-transpose
```

Hydra override syntax examples:
```
train optimizer.lr=0.0012                          # override the default for a value in the config dataclass
train optimizer=adamw +optimizer.amsgrad=True      # add a new value that wasn't tracked in the config dataclass
```

If you want to debug the configuration, add `-c job` to the end of your command. It will print the Hydra config instead
of running the job.

You can see the available config options to choose from by running `train --help`. The config options are also shown
below for convenience.

```
== Configuration groups ==
Compose your configuration from those groups (group=option)

collator: multi-seq-dict
compute: a100, a10g, cpu, local
dataset: bach, maestro, nes, scales, symphony-net
logger: tensorboard, wandb, wandb-test
lr_scheduler: cosine, plateau
model: mmt, multihead-transformer, structured
network: mmt, mmt-medium, mmt-small, structured, structured-medium, structured-small, tpd, tpd-medium, tpd-small
optimizer: adam, adamw, sgd
tokenizer: mmt, structured, tpd
trainer: cpu, gpu, mps, smoke-test
transforms: crop, crop-transpose
```
