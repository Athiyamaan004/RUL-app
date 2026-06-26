import numpy as np
import pandas as pd

from scipy.stats import kurtosis
from scipy.stats import skew

def extract_vibration_features(csv_file):

    df = pd.read_csv(
        csv_file,
        header=None
    )

    horizontal = pd.to_numeric(
        df.iloc[:,4],
        errors="coerce"
    ).dropna()

    vertical = pd.to_numeric(
        df.iloc[:,5],
        errors="coerce"
    ).dropna()

    rms_h = np.sqrt(
        np.mean(horizontal**2)
    )

    rms_v = np.sqrt(
        np.mean(vertical**2)
    )

    feat = {}

    feat["rms_h"] = rms_h
    feat["rms_v"] = rms_v

    feat["kurtosis_h"] = kurtosis(horizontal)
    feat["kurtosis_v"] = kurtosis(vertical)

    feat["std_h"] = np.std(horizontal)
    feat["std_v"] = np.std(vertical)

    feat["mean_h"] = np.mean(horizontal)
    feat["mean_v"] = np.mean(vertical)

    feat["peak_to_peak_h"] = np.max(horizontal) - np.min(horizontal)
    feat["peak_to_peak_v"] = np.max(vertical) - np.min(vertical)

    feat["skew_h"] = skew(horizontal)
    feat["skew_v"] = skew(vertical)

    feat["var_h"] = np.var(horizontal)
    feat["var_v"] = np.var(vertical)

    feat["peak_h"] = np.max(np.abs(horizontal))
    feat["peak_v"] = np.max(np.abs(vertical))

    feat["crest_h"] = (
        feat["peak_h"] / rms_h
    )

    feat["crest_v"] = (
        feat["peak_v"] / rms_v
    )

    fft_h = np.abs(
        np.fft.rfft(horizontal)
    )

    feat["spec_energy_h"] = np.sum(
        fft_h**2
    )

    feat["peak_freq_amp_h"] = np.max(
        fft_h
    )

    feat["mean_fft_h"] = np.mean(
        fft_h
    )

    freqs_h = np.fft.rfftfreq(
        len(horizontal),
        d=1/25600
    )

    feat["spectral_centroid_h"] = (
        np.sum(freqs_h * fft_h)
        /
        np.sum(fft_h)
    )

    fft_v = np.abs(
        np.fft.rfft(vertical)
    )

    feat["spec_energy_v"] = np.sum(
        fft_v**2
    )

    feat["peak_freq_amp_v"] = np.max(
        fft_v
    )

    feat["mean_fft_v"] = np.mean(
        fft_v
    )

    freqs_v = np.fft.rfftfreq(
        len(vertical),
        d=1/25600
    )

    feat["spectral_centroid_v"] = (
        np.sum(freqs_v * fft_v)
        /
        np.sum(fft_v)
    )

    return feat
