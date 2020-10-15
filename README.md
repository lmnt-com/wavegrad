# WaveGrad
![PyPI Release](https://img.shields.io/pypi/v/wavegrad?label=release) [![License](https://img.shields.io/github/license/lmnt-com/wavegrad)](https://github.com/lmnt-com/wavegrad/blob/master/LICENSE)

WaveGrad is a fast, high-quality neural vocoder designed by the folks at Google Brain. The architecture is described in [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf). In short, this model takes a log-scaled Mel spectrogram and converts it to a waveform via iterative refinement.

## Status (2020-10-15)
- [x] stable training (22 kHz, 24 kHz)
- [x] high-quality synthesis
- [x] mixed-precision training
- [x] multi-GPU training
- [x] custom noise schedule (faster inference)
- [x] command-line inference
- [x] programmatic inference API
- [x] PyPI package
- [x] audio samples
- [x] pretrained models
- [ ] precomputed noise schedule

## Audio samples
[24 kHz audio samples](https://lmnt.com/assets/wavegrad/24kHz)

## Pretrained models
[24 kHz pretrained model](https://lmnt.com/assets/wavegrad/wavegrad-24kHz.pt) (183 MB, SHA256: `65e9366da318d58d60d2c78416559351ad16971de906e53b415836c068e335f3`)

## Install

Install using pip:
```
pip install wavegrad
```

or from GitHub:
```
git clone https://github.com/lmnt-com/wavegrad.git
cd wavegrad
pip install .
```

### Training
Before you start training, you'll need to prepare a training dataset. The dataset can have any directory structure as long as the contained .wav files are 16-bit mono (e.g. [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), [VCTK](https://pytorch.org/audio/_modules/torchaudio/datasets/vctk.html)). By default, this implementation assumes a sample rate of 22 kHz. If you need to change this value, edit [params.py](https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/params.py).

```
python -m wavegrad.preprocess /path/to/dir/containing/wavs
python -m wavegrad /path/to/model/dir /path/to/dir/containing/wavs

# in another shell to monitor training progress:
tensorboard --logdir /path/to/model/dir --bind_all
```

You should expect to hear intelligible speech by ~20k steps (~1.5h on a 2080 Ti).

### Inference API
Basic usage:

```python
from wavegrad.inference import predict as wavegrad_predict

model_dir = '/path/to/model/dir'
spectrogram = # get your hands on a spectrogram in [N,C,W] format
audio, sample_rate = wavegrad_predict(spectrogram, model_dir)

# audio is a GPU tensor in [N,T] format.
```

If you have a custom noise schedule (see below):
```python
from wavegrad.inference import predict as wavegrad_predict

params = { 'noise_schedule': np.load('/path/to/noise_schedule.npy') }
model_dir = '/path/to/model/dir'
spectrogram = # get your hands on a spectrogram in [N,C,W] format
audio, sample_rate = wavegrad_predict(spectrogram, model_dir, params=params)

# `audio` is a GPU tensor in [N,T] format.
```

### Inference CLI
```
python -m wavegrad.inference /path/to/model /path/to/spectrogram -o output.wav
```

### Noise schedule
The default implementation uses 1000 iterations to refine the waveform, which runs slower than real-time. WaveGrad is able to achieve high-quality, faster than real-time synthesis with as few as 6 iterations without re-training the model with new hyperparameters.

To achieve this speed-up, you will need to search for a `noise schedule` that works well for your dataset. This implementation provides a script to perform the search for you:

```
python -m wavegrad.noise_schedule /path/to/trained/model /path/to/preprocessed/validation/dataset
python -m wavegrad.inference /path/to/trained/model /path/to/spectrogram -n noise_schedule.npy -o output.wav
```

The default settings should give good results without spending too much time on the search. If you'd like to find a better noise schedule or use a different number of inference iterations, run the `noise_schedule` script with `--help` to see additional configuration options.


## References
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/pdf/2009.00713.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
