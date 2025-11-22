# Time Series Domains 101: Synthetic examples + visualizations
#
# This notebook cell:
# 1) Creates a simple "taxonomy" diagram image of time-series domains.
# 2) Generates five synthetic time series (stationary, trending, periodic, aperiodic/changing-frequency, and noisy).
# 3) For each series, renders:
#    - Time-domain plot
#    - Frequency-domain (FFT magnitude) plot
#    - Time–frequency view (spectrogram via STFT)
#    - Lag-domain view (autocorrelation function, ACF)
#
# Notes:
# - Uses matplotlib only (no seaborn), one chart per figure (no subplots), and default colors.
# - Saves figures to /mnt/data so you can download them if desired.
#
import numpy as np
import matplotlib.pyplot as plt
import os
from textwrap import dedent

np.random.seed(42)

# ---------- Utility: ensure output dir ----------
out_dir = "/mnt/data/ts_domains_demo"
os.makedirs(out_dir, exist_ok=True)

# ---------- 0) Draw a simple taxonomy diagram ----------
def draw_taxonomy_diagram(filepath):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Boxes as (x, y, width, height, label)
    boxes = [
        (0.05, 0.75, 0.35, 0.18, "Time Domain\n(values vs. time)\n• Trends\n• Local shape\n• Autocorrelation"),
        (0.60, 0.75, 0.35, 0.18, "Frequency Domain\n(values vs. frequency)\n• Periodicity\n• Spectral power"),
        (0.05, 0.45, 0.35, 0.18, "Time–Frequency\n(when + what freq)\n• Spectrogram (STFT)\n• Wavelets"),
        (0.60, 0.45, 0.35, 0.18, "Lag Domain\n(values vs. lag)\n• ACF/PACF\n• Dependency @ delays"),
        (0.05, 0.15, 0.35, 0.18, "State-Space\n(latent dynamics)\n• Kalman/SSMs\n• Neural SSMs"),
        (0.60, 0.15, 0.35, 0.18, "Spatio-Temporal\n(space + time)\n• Graph models\n• VAR across nodes"),
    ]

    for (x, y, w, h, label) in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10)

    # Arrows to suggest complementarity
    arrow_pairs = [
        ((0.225, 0.75), (0.775, 0.75)),  # time -> frequency
        ((0.225, 0.64), (0.225, 0.54)),  # time -> time-freq
        ((0.775, 0.64), (0.775, 0.54)),  # freq -> lag
        ((0.225, 0.34), (0.225, 0.25)),  # time-freq -> state-space
        ((0.775, 0.34), (0.775, 0.25)),  # lag -> spatio-temporal
    ]
    for (x0, y0), (x1, y1) in arrow_pairs:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->"))

    ax.set_title("Time-Series Domains: A Quick Map", fontsize=14)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)

taxonomy_path = os.path.join(out_dir, "time_series_taxonomy.png")
draw_taxonomy_diagram(taxonomy_path)

# ---------- 1) Generate synthetic series ----------
N = 2048  # length of each series (power of 2 helps FFT, but not required)
t = np.linspace(0, 1, N, endpoint=False)

def series_stationary_ar1(phi=0.7, sigma=0.5):
    x = np.zeros(N)
    eps = np.random.normal(scale=sigma, size=N)
    for i in range(1, N):
        x[i] = phi * x[i-1] + eps[i]
    return x

def series_trending_linear_sine():
    trend = 5 * t  # linear upward trend
    seasonal = 0.8 * np.sin(2 * np.pi * 6 * t)  # mild seasonality
    noise = 0.3 * np.random.normal(size=N)
    return trend + seasonal + noise

def series_periodic_sine():
    s = 2.0 * np.sin(2 * np.pi * 12 * t)  # clear periodic wave
    s += 0.2 * np.random.normal(size=N)
    return s

def series_aperiodic_changing_freq():
    # piecewise frequencies: 0-1/3: f=6, 1/3-2/3: f=18, 2/3-1: f=9
    s = np.zeros(N)
    idx1 = int(N/3); idx2 = int(2*N/3)
    s[:idx1] = np.sin(2 * np.pi * 6 * t[:idx1])
    s[idx1:idx2] = np.sin(2 * np.pi * 18 * t[idx1:idx2])
    s[idx2:] = np.sin(2 * np.pi * 9 * t[idx2:])
    s += 0.2 * np.random.normal(size=N)
    return s

def series_white_noise():
    return np.random.normal(scale=1.0, size=N)

series_dict = {
    "stationary_ar1": series_stationary_ar1(),
    "trending": series_trending_linear_sine(),
    "periodic": series_periodic_sine(),
    "aperiodic_changing_freq": series_aperiodic_changing_freq(),
    "noisy_white": series_white_noise(),
}

# ---------- 2) Helpers: FFT magnitude, spectrogram, and ACF ----------
def fft_mag(x, fs=1.0):
    # Return positive-frequency magnitude spectrum
    X = np.fft.rfft(x)  # real FFT: N//2+1 bins
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    mag = np.abs(X) / len(x)
    return freqs, mag

def acf(x, max_lag=200):
    x = np.asarray(x)
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size//2:]  # keep non-negative lags
    corr /= corr[0]  # normalize
    lags = np.arange(len(corr))
    return lags[:max_lag+1], corr[:max_lag+1]

def plot_and_save(y, title, xlabel, ylabel, filepath):
    plt.figure(figsize=(8, 3))
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()

def plot_time_series(x, name):
    path = os.path.join(out_dir, f"{name}__time.png")
    plot_and_save(x, f"{name}: time domain", "time (index)", "value", path)
    return path

def plot_fft(x, name, fs=1.0):
    f, m = fft_mag(x, fs=fs)
    plt.figure(figsize=(8, 3))
    plt.plot(f, m)
    plt.title(f"{name}: frequency domain (FFT magnitude)")
    plt.xlabel("frequency (cycles per unit time)")
    plt.ylabel("|X(f)|")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}__fft.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path

def plot_spectrogram(x, name, fs=1.0, nperseg=256, noverlap=128):
    # Use matplotlib.specgram for STFT-like view
    plt.figure(figsize=(8, 3))
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=nperseg, Fs=fs, noverlap=noverlap)
    plt.title(f"{name}: time–frequency (spectrogram)")
    plt.xlabel("time (bins)")
    plt.ylabel("frequency")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}__spectrogram.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path

def plot_acf_fig(x, name, max_lag=200):
    lags, c = acf(x, max_lag=max_lag)
    plt.figure(figsize=(8, 3))
    plt.stem(lags, c, use_line_collection=True)
    plt.title(f"{name}: lag domain (ACF)")
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}__acf.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path

# ---------- 3) Generate and save all plots ----------
generated_files = []

# taxonomy diagram
generated_files.append(taxonomy_path)

for name, x in series_dict.items():
    generated_files.append(plot_time_series(x, name))
    generated_files.append(plot_fft(x, name, fs=1.0))
    generated_files.append(plot_spectrogram(x, name, fs=1.0, nperseg=256, noverlap=128))
    generated_files.append(plot_acf_fig(x, name, max_lag=200))

# Provide a short text summary file listing outputs
summary_txt = os.path.join(out_dir, "README.txt")
with open(summary_txt, "w") as f:
    f.write(dedent(f"""
    Time-Series Domains Demo
    ------------------------
    Generated files for each synthetic series under: {out_dir}

    Series:
      - stationary_ar1
      - trending
      - periodic
      - aperiodic_changing_freq
      - noisy_white

    For each series, the following views are saved:
      - __time.png           : Time domain
      - __fft.png            : Frequency domain (FFT magnitude)
      - __spectrogram.png    : Time–frequency domain (STFT-based spectrogram)
      - __acf.png            : Lag domain (ACF)

    Taxonomy diagram:
      - time_series_taxonomy.png
    """))

generated_files.append(summary_txt)

generated_files
