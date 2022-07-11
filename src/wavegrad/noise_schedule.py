# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from glob import glob
from itertools import product as cartesian_product
from tqdm import tqdm

from wavegrad.params import params
from wavegrad.inference import predict


def _round_up(x, multiple):
  return (x + multiple - 1) // multiple * multiple


def _ls_mse(reference, predicted):
  sr = params.sample_rate
  hop = params.hop_samples // 2
  win = params.hop_samples * 4
  n_fft = 2**((win-1).bit_length())
  f_max = sr / 2.0
  mel_spec_transform = TT.MelSpectrogram(
      sample_rate=params.sample_rate,
      n_fft=n_fft,
      win_length=win,
      hop_length=hop,
      f_min=20.0,
      f_max=f_max,
      power=1.0,
      normalized=True).cuda()

  reference = torch.log(mel_spec_transform(reference) + 1e-5)
  predicted = torch.log(mel_spec_transform(predicted) + 1e-5)
  return torch.sum((reference - predicted)**2)


def main(args):
  audio_filenames = glob(f'{args.data_dir}/**/*.wav', recursive=True)
  audio_filenames = [f for f in audio_filenames if os.path.exists(f'{f}.spec.npy')]
  if len(audio_filenames) == 0:
    raise ValueError('No files found.')

  audio = []
  spectrogram = []
  max_audio_len = 0
  max_spec_len = 0
  for filename in audio_filenames[:args.batch_size]:
    clip, _ = T.load(filename)
    clip = clip[0].numpy()
    spec = np.load(f'{filename}.spec.npy')
    audio.append(clip)
    spectrogram.append(spec)
    max_audio_len = max(max_audio_len, _round_up(len(clip), params.hop_samples))
    max_spec_len = max(max_spec_len, spec.shape[1])

  padded_audio = [np.pad(a, [0, max_audio_len - len(a)], mode='constant') for a in audio]
  spectrogram = [np.pad(s, [[0, 0], [0, max_spec_len - s.shape[1]]], mode='constant') for s in spectrogram]

  padded_audio = torch.from_numpy(np.stack(padded_audio)).cuda()
  spectrogram = torch.from_numpy(np.stack(spectrogram)).cuda()

  mantissa = list(sorted(10 * np.random.uniform(size=args.search_level)))
  exponent = 10**np.linspace(-6, -1, num=args.iterations)
  best_score = 1e32
  for candidate in tqdm(cartesian_product(mantissa, repeat=args.iterations), total=len(mantissa)**args.iterations):
    noise_schedule = np.array(candidate) * exponent
    predicted, _ = predict(spectrogram, model_dir=args.model_dir, params={ 'noise_schedule': noise_schedule })
    score = _ls_mse(padded_audio, predicted)
    if score < best_score:
      best_score = score
      np.save(args.output, noise_schedule)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs a search to find the best noise schedule for a specified number of inference iterations')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('data_dir',
      help='directory from which to read .wav and spectrogram files for noise schedule opitimization')
  parser.add_argument('--batch-size', '-b', type=int, default=1,
      help='how many wav files to use for the optimization process')
  parser.add_argument('--iterations', '-i', type=int, default=6,
      help='how many refinement steps to use during inference (more => increase inference time)')
  parser.add_argument('--search-level', '-s', type=int, default=3, choices=range(1, 20),
      help='how many points to use for the search (more => exponentially more time to search)')
  parser.add_argument('--output', '-o', default='noise_schedule.npy',
      help='output file name')
  main(parser.parse_args())
