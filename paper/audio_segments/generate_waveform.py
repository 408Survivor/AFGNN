"""Generate waveform and spectrogram plots from an audio segment."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
input_wav = "paper/audio_segments/segment_9min.wav"
out_dir = "paper/audio_segments"

# ---------------------------------------------------------------------------
# Load audio
# ---------------------------------------------------------------------------
sr, data = wavfile.read(input_wav)
if data.ndim > 1:
    data = data.mean(axis=1)  # mono fallback

data = data.astype(np.float32)
# Normalize to [-1, 1]
max_amp = np.max(np.abs(data))
if max_amp != 0:
    data = data / max_amp

t = np.arange(len(data)) / sr

# ---------------------------------------------------------------------------
# Plot 1: clean decorative waveform (suitable for framework figure)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 1.2), dpi=150)
ax.fill_between(t, data, color="#4A90A4", alpha=0.85)
ax.plot(t, data, color="#2E5A6B", linewidth=0.6)
ax.set_xlim(t.min(), t.max())
ax.set_ylim(-1.05, 1.05)
ax.axis("off")  # no axes for decorative use
fig.patch.set_alpha(0)
plt.tight_layout(pad=0)
plt.savefig(os.path.join(out_dir, "waveform_9min.png"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.savefig(os.path.join(out_dir, "waveform_9min.pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: waveform with axis labels (for inspection / standalone)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 2), dpi=150)
ax.fill_between(t, data, color="#4A90A4", alpha=0.7)
ax.plot(t, data, color="#2E5A6B", linewidth=0.5)
ax.set_xlim(t.min(), t.max())
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel("Amplitude", fontsize=10)
ax.set_title("Extracted audio waveform (5 s segment)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "waveform_9min_with_axes.png"), dpi=150)
plt.savefig(os.path.join(out_dir, "waveform_9min_with_axes.pdf"))
plt.close()

# ---------------------------------------------------------------------------
# Plot 3: clean decorative spectrogram (suitable for framework figure)
# ---------------------------------------------------------------------------
# Use mel-like perceptual scaling via simple log-frequency mapping
n_fft = 512
hop_length = n_fft // 4
f, t_spec, Zxx = stft(data, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
S = np.abs(Zxx)
# Keep only frequencies up to ~8 kHz for speech-focused visualization
f_max_idx = np.argmax(f >= 8000)
if f_max_idx == 0:
    f_max_idx = len(f)
f = f[:f_max_idx]
S = S[:f_max_idx, :]
S_db = 20 * np.log10(S + 1e-10)

fig, ax = plt.subplots(figsize=(6, 1.8), dpi=150)
ax.imshow(
    S_db,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    extent=[t_spec.min(), t_spec.max(), f.min() / 1000, f.max() / 1000],
)
ax.axis("off")
fig.patch.set_alpha(0)
plt.tight_layout(pad=0)
plt.savefig(os.path.join(out_dir, "spectrogram_9min.png"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.savefig(os.path.join(out_dir, "spectrogram_9min.pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.close()

# ---------------------------------------------------------------------------
# Plot 4: spectrogram with axis labels (for inspection / standalone)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 2.5), dpi=150)
im = ax.imshow(
    S_db,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    extent=[t_spec.min(), t_spec.max(), f.min() / 1000, f.max() / 1000],
)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel("Frequency (kHz)", fontsize=10)
ax.set_title("Spectrogram (5 s segment)", fontsize=11)
cbar = plt.colorbar(im, ax=ax, format="%d dB")
cbar.set_label("Power (dB)", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "spectrogram_9min_with_axes.png"), dpi=150)
plt.savefig(os.path.join(out_dir, "spectrogram_9min_with_axes.pdf"))
plt.close()

print("Saved files in", out_dir)
print("  waveform_9min.png/pdf")
print("  waveform_9min_with_axes.png/pdf")
print("  spectrogram_9min.png/pdf")
print("  spectrogram_9min_with_axes.png/pdf")
print(f"Duration: {len(data)/sr:.2f}s, sample rate: {sr} Hz")
