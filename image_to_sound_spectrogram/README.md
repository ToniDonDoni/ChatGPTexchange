# Image to sound spectrogram experiment

This note describes the idea behind converting an image into an audio file so that the original image becomes visible when the audio is viewed as a spectrogram.

This is not byte-perfect image archiving and not a modem protocol. It is closer to spectrogram art: the horizontal axis of the image becomes time, the vertical axis becomes frequency, and pixel brightness becomes the loudness of the corresponding tone.

## What the pipeline does

1. Start with the source image.
2. Crop away irrelevant background and keep the main subject.
3. Convert the image to grayscale.
4. Enhance edges with a Sobel filter so important shapes remain readable in the spectrogram.
5. Resize the image to a `time_frames x frequency_bins` grid.
6. Map the vertical axis into an audible frequency range, for example `320 Hz .. 4800 Hz`.
7. For each time frame, create a frequency spectrum where bright pixels produce louder tones and dark pixels are nearly silent.
8. Use inverse FFT to synthesize the audio waveform.
9. Normalize the loudness and optionally encode the result as MP3.

## What it sounds like

The result sounds like digital tones, noise, and chirping. That is expected. The audio is not meant to be musical. The point is that the image can be seen when the file is opened in a spectrogram viewer.

## Minimal Python sketch

```python
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage, signal
from scipy.io import wavfile

src = "input.jpg"
out = "output.wav"

sr = 44100
duration = 13.0
n_fft = 4096
n_frames = 520
hop = int(round((duration * sr - n_fft) / (n_frames - 1)))
length = n_fft + hop * (n_frames - 1)

f_min, f_max = 320, 4800
freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
bins = np.where((freqs >= f_min) & (freqs <= f_max))[0]

img = Image.open(src).convert("RGB")
gray = ImageOps.grayscale(img)
gray = ImageOps.autocontrast(gray)
gray = gray.resize((n_frames, len(bins)), Image.Resampling.LANCZOS)
g = np.asarray(gray, dtype=np.float32) / 255.0

# Add edges so the shape remains readable.
sx = ndimage.sobel(g, axis=1)
sy = ndimage.sobel(g, axis=0)
edges = np.hypot(sx, sy)
edges /= edges.max() + 1e-9

amp = 0.4 * g + 0.6 * edges
amp = np.clip(amp, 0, 1)
amp = amp[::-1, :]  # spectrogram frequency bins go from low to high

rng = np.random.default_rng(42)
phase = rng.uniform(0, 2 * np.pi, size=len(freqs))
phase_inc = 2 * np.pi * freqs * hop / sr
window = signal.windows.hann(n_fft, sym=False)

y = np.zeros(length)
win_sum = np.zeros(length)

for t in range(n_frames):
    mag = np.zeros(len(freqs))
    mag[bins] = amp[:, t]
    spectrum = mag * np.exp(1j * phase)
    frame = np.fft.irfft(spectrum, n=n_fft)

    start = t * hop
    y[start:start + n_fft] += frame * window
    win_sum[start:start + n_fft] += window ** 2
    phase = (phase + phase_inc) % (2 * np.pi)

mask = win_sum > 1e-8
y[mask] /= np.sqrt(win_sum[mask])
y -= y.mean()
y /= np.max(np.abs(y)) + 1e-9
y *= 0.7

wavfile.write(out, sr, np.int16(y * 32767))
```

## How to view the hidden image

Open the WAV or MP3 in any audio tool with a spectrogram view:

- Audacity: switch the track view to `Spectrogram`.
- Sonic Visualiser.
- Python with `scipy.signal.stft` and `matplotlib.imshow`.

## Important limitation

MP3 damages the spectrogram more than WAV because MP3 is lossy compression. For the cleanest visible image, store the result as WAV or FLAC. MP3 is fine for a small demo, but it can blur fine details.

## If byte-perfect recovery is needed

Use a different approach: encode the image bytes as an audio modem signal, for example FSK, PSK, or QAM, with a header, checksum, and error correction. That is data transmission through audio, not spectrogram art.
