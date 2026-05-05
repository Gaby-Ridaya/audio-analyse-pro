#!/usr/bin/env python3
"""
Analyse audio professionnelle orientee mixage et mastering studio.
- LUFS robuste (mono/stereo)
- Spectral centroid, rolloff, flatness, contrast
- Loudness Range (LRA)
- True Peak via oversampling x4
- Analyse Mid/Side complete avec ratio dB
- Analyse interpretation et conseils mixage
- Comparaison reference (compare_reference)
- STFT optimise (calcul unique)
- Export JSON + CSV
"""

import librosa
import librosa.feature
import librosa.beat
import librosa.onset
import numpy as np
import pyloudnorm as pyln  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]
import json
import os
import argparse
import csv
import gc
from scipy.signal import resample_poly
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import matplotlib.pyplot as plt
from typing import Any
from numpy.typing import NDArray

console = Console()

# ─────────────────────────────────────────────
# Constantes DSP
# ─────────────────────────────────────────────
N_FFT = 2048
HOP_LENGTH = 512
TRUE_PEAK_OVERSAMPLE = 4  # facteur de surechantillonnage pour True Peak


def detect_key(y: NDArray[Any], sr: int | float) -> str:
    """Detection de la tonalite via chroma CQT."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return notes[np.argmax(chroma_mean)]
    except Exception:
        return "Indeterminee"


def compute_stft(y: NDArray[Any], sr: int | float,
                 n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> tuple:
    """
    Calcul unique de la STFT. Reutilise pour toutes les features spectrales.
    Retourne (S_magnitude, frequencies). S stockee en float32 pour economiser la RAM.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S, frequencies


def compute_lufs_robust(data: NDArray[Any], rate: int | float) -> float:
    """
    Calcul LUFS robuste : gere mono, stereo, et multicanal.
    Accepte un tableau numpy (samples,) ou (samples, channels).
    """
    try:
        meter = pyln.Meter(rate)
        d = data
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        elif d.ndim == 2 and d.shape[1] > 2:
            d = d[:, :2]
        lufs = float(meter.integrated_loudness(d.astype(np.float64)))
        return lufs if np.isfinite(lufs) else -99.0
    except Exception:
        return -99.0


def compute_true_peak(data: NDArray[Any], factor: int = TRUE_PEAK_OVERSAMPLE,
                      chunk_sec: float = 30.0, sr: int = 44100) -> float:
    """
    True Peak via surechantillonnage (oversampling x4), traitement par blocs
    pour eviter l'explosion memoire sur les longs fichiers.
    Conforme a ITU-R BS.1770.
    """
    try:
        chunk_size = int(chunk_sec * sr)
        n_samples = data.shape[0] if data.ndim == 1 else data.shape[0]
        n_channels = 1 if data.ndim == 1 else data.shape[1]
        peak = 0.0
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = data[start:end] if data.ndim == 1 else data[start:end, :]
            if data.ndim == 1:
                upsampled = resample_poly(chunk, factor, 1)
                peak = max(peak, float(np.max(np.abs(upsampled))))
            else:
                for ch in range(n_channels):
                    upsampled = resample_poly(chunk[:, ch], factor, 1)
                    peak = max(peak, float(np.max(np.abs(upsampled))))
        return peak
    except Exception:
        return float(np.max(np.abs(data)))


def compute_true_peak_db(data: NDArray[Any], factor: int = TRUE_PEAK_OVERSAMPLE) -> float:
    """True Peak en dBTP."""
    tp = compute_true_peak(data, factor)
    return round(20 * np.log10(tp + 1e-12), 2)


def compute_loudness_range(data: NDArray[Any], rate: int | float) -> float:
    """
    Loudness Range (LRA) en LU.
    Accepte un tableau numpy pre-charge (samples,) ou (samples, channels).
    """
    try:
        d = data
        if d.ndim == 1:
            d = d.reshape(-1, 1)

        meter = pyln.Meter(rate)

        # Fenetre de 3 secondes, hop de 1 seconde (ITU-R BS.1770)
        window_samples = int(3.0 * rate)
        hop_samples = int(1.0 * rate)
        n_samples = d.shape[0]

        if n_samples < window_samples:
            return 0.0

        short_term_loudness = []
        pos = 0
        while pos + window_samples <= n_samples:
            # Conversion float64 par bloc uniquement (evite copie complete)
            block = d[pos:pos + window_samples].astype(np.float64)
            try:
                loud = meter.integrated_loudness(block)
                if np.isfinite(loud) and loud > -70.0:
                    short_term_loudness.append(loud)
            except Exception:
                pass
            pos += hop_samples

        if len(short_term_loudness) < 4:
            return 0.0

        arr = np.array(short_term_loudness)
        # LRA = difference entre percentile 95 et percentile 10
        lra = float(np.percentile(arr, 95) - np.percentile(arr, 10))
        return round(lra, 2)
    except Exception:
        return 0.0


def compute_mid_side(data_raw: NDArray[Any]) -> dict:
    """
    Analyse Mid/Side complete.
    Mid = (L + R) / 2, Side = (L - R) / 2.
    Accepte un tableau numpy (samples, channels) pre-charge.
    """
    try:
        if data_raw.ndim != 2 or data_raw.shape[1] < 2:
            return {
                "is_stereo": False,
                "mid_rms": 0.0,
                "side_rms": 0.0,
                "ms_ratio_db": 0.0,
                "correlation": 1.0,
                "stereo_width": 0.0,
            }

        left = data_raw[:, 0]
        right = data_raw[:, 1]
        mid = (left + right) / 2.0
        side = (left - right) / 2.0

        mid_rms = float(np.sqrt(np.mean(mid.astype(np.float64) ** 2)))
        side_rms = float(np.sqrt(np.mean(side.astype(np.float64) ** 2)))

        # Ratio M/S en dB
        ms_ratio_db = 20 * np.log10((side_rms + 1e-12) / (mid_rms + 1e-12))

        # Correlation inter-canal (calcul manuel en float32 pour eviter copie float64)
        lm = left - float(np.mean(left))
        rm = right - float(np.mean(right))
        num = float(np.sum(lm * rm))
        den = float(np.sqrt(np.sum(lm ** 2) * np.sum(rm ** 2)))
        correlation = num / (den + 1e-12)
        stereo_width = 1.0 - abs(correlation)

        return {
            "is_stereo": True,
            "mid_rms": round(mid_rms, 6),
            "side_rms": round(side_rms, 6),
            "ms_ratio_db": round(float(ms_ratio_db), 2),
            "correlation": round(correlation, 4),
            "stereo_width": round(stereo_width, 4),
        }
    except Exception:
        return {
            "is_stereo": False,
            "mid_rms": 0.0,
            "side_rms": 0.0,
            "ms_ratio_db": 0.0,
            "correlation": 1.0,
            "stereo_width": 0.0,
        }


def _kweighting_filters(sr: float) -> tuple:
    """
    Coefficients des filtres K-weighting (ITU-R BS.1770).
    Stage 1 : pre-filtre high-shelf.
    Stage 2 : ponderation high-pass RLB.
    """
    from scipy.signal import lfilter_zi
    # Stage 1 : pre-filtre high-shelf
    f0, G, Q = 1681.974450955533, 3.99984385397, 0.7071752369554196
    K1 = np.tan(np.pi * f0 / sr)
    Vh = 10.0 ** (G / 20.0)
    Vb = 10.0 ** ((G / 2.0) / 20.0)
    n1 = 1.0 + K1 / Q + K1 * K1
    b1 = np.array([(Vh + Vb * K1 / Q + K1 * K1) / n1,
                   2.0 * (K1 * K1 - Vh) / n1,
                   (Vh - Vb * K1 / Q + K1 * K1) / n1])
    a1 = np.array([1.0, 2.0 * (K1 * K1 - 1.0) / n1,
                   (1.0 - K1 / Q + K1 * K1) / n1])
    # Stage 2 : high-pass RLB
    f0, Q = 38.13547087602444, 0.5003270373238773
    K2 = np.tan(np.pi * f0 / sr)
    n2 = 1.0 + K2 / Q + K2 * K2
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (K2 * K2 - 1.0) / n2,
                   (1.0 - K2 / Q + K2 * K2) / n2])
    # Etats initiaux
    zi1 = lfilter_zi(b1, a1)
    zi2 = lfilter_zi(b2, a2)
    return b1, a1, b2, a2, zi1, zi2


def _gated_lufs(block_powers: list) -> float:
    """Calcul LUFS integree avec double gating (ITU-R BS.1770-4)."""
    if not block_powers:
        return -99.0
    p = np.array(block_powers)
    # Gate absolue a -70 LUFS
    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    p_abs = p[p > abs_gate]
    if len(p_abs) == 0:
        return -70.0
    lufs_g = -0.691 + 10.0 * np.log10(np.mean(p_abs))
    # Gate relative a lufs_g - 10 LU
    rel_gate = 10.0 ** ((lufs_g - 10.0 + 0.691) / 10.0)
    p_rel = p_abs[p_abs > rel_gate]
    if len(p_rel) == 0:
        return float(lufs_g)
    return float(-0.691 + 10.0 * np.log10(np.mean(p_rel)))


def _stream_load_mono(fichier: str, target_sr: int = 22050,
                      block_sec: float = 30.0) -> tuple:
    """
    Charge un fichier audio en blocs de 30s (streaming).
    Calcule LUFS et LRA via K-weighting IIR stateful a la volee.
    Pic memoire par bloc : ~25 MB. Retourne (y_mono, sr, rms, peak, lufs, lra, ms_dict).
    """
    from scipy.signal import lfilter
    info = sf.info(fichier)
    native_sr = info.samplerate
    n_channels = info.channels
    block_size = int(block_sec * native_sr)

    # ── Filtres K-weighting pour LUFS/LRA a native_sr ──
    b1, a1, b2, a2, zi1_t, zi2_t = _kweighting_filters(native_sr)
    # Un etat IIR par canal (max 2 canaux)
    n_ch = min(n_channels, 2)
    zi1 = [zi1_t.copy() for _ in range(n_ch)]
    zi2 = [zi2_t.copy() for _ in range(n_ch)]

    # Accumulateurs LUFS (400ms blocs, 75% overlap)
    lufs_block_n = int(0.400 * native_sr)
    lufs_stride = int(0.100 * native_sr)
    # Accumulateurs LRA (3s blocs, 1s hop)
    lra_block_n = int(3.0 * native_sr)
    lra_stride = int(1.0 * native_sr)

    lufs_powers: list = []
    lra_powers: list = []
    kw_leftover = np.empty(0, dtype=np.float64)  # reste K-weighted inter-blocs

    # Accumulateurs RMS/peak/M/S
    y_blocks: list = []
    sum_sq = 0.0
    n_total = 0
    peak = 0.0
    n_ms = 0
    sum_l = sum_r = sum_l2 = sum_r2 = sum_lr = 0.0
    sum_m2 = sum_s2 = 0.0

    with sf.SoundFile(fichier) as f:
        for block in f.blocks(blocksize=block_size, dtype='float32'):
            # ── K-weighting stateful sur block stereo natif ──
            block_f64 = block.reshape(-1, n_ch).astype(np.float64)
            kw_ch = np.zeros_like(block_f64)
            for ch in range(n_ch):
                s1_out, zi1[ch] = lfilter(b1, a1, block_f64[:, ch], zi=zi1[ch])
                kw_ch[:, ch], zi2[ch] = lfilter(b2, a2, s1_out, zi=zi2[ch])
            # Somme quadratique des canaux (canal unique si mono)
            kw_mono_f64 = np.mean(kw_ch ** 2, axis=1)
            del block_f64, kw_ch

            # ── Ajouter le reste du bloc precedent ──
            kw_full = np.concatenate([kw_leftover, kw_mono_f64])

            # ── Extraire blocs LUFS 400ms ──
            pos = 0
            while pos + lufs_block_n <= len(kw_full):
                lufs_powers.append(float(np.mean(kw_full[pos:pos + lufs_block_n])))
                pos += lufs_stride
            # ── Extraire blocs LRA 3s ──
            pos_lra = 0
            while pos_lra + lra_block_n <= len(kw_full):
                lra_powers.append(float(np.mean(kw_full[pos_lra:pos_lra + lra_block_n])))
                pos_lra += lra_stride
            kw_leftover = kw_full[pos:]  # reste pour le prochain bloc
            del kw_full, kw_mono_f64

            # ── Mid/Side depuis bloc stereo (formule incrementale) ──
            if block.ndim == 2 and block.shape[1] >= 2:
                l = block[:, 0].astype(np.float64)
                r = block[:, 1].astype(np.float64)
                n_ms += len(l)
                sum_l += float(np.sum(l))
                sum_r += float(np.sum(r))
                sum_l2 += float(np.sum(l * l))
                sum_r2 += float(np.sum(r * r))
                sum_lr += float(np.sum(l * r))
                m = (l + r) / 2.0
                s = (l - r) / 2.0
                sum_m2 += float(np.sum(m * m))
                sum_s2 += float(np.sum(s * s))
                del l, r, m, s
                mono = block.mean(axis=1).astype(np.float32)
            else:
                mono = block.ravel().astype(np.float32)

            # ── Resample mono vers target_sr ──
            if native_sr != target_sr:
                mono = librosa.resample(mono, orig_sr=native_sr, target_sr=target_sr)

            sum_sq += float(np.sum(mono.astype(np.float64) ** 2))
            n_total += len(mono)
            peak = max(peak, float(np.max(np.abs(mono))))
            y_blocks.append(mono)

    # ── Concatenation finale (pic = 2x taille de y) ──
    y = np.concatenate(y_blocks)
    del y_blocks
    gc.collect()

    rms = float(np.sqrt(sum_sq / max(n_total, 1)))

    # ── LUFS integree (ITU-R BS.1770-4 gated) ──
    lufs = _gated_lufs(lufs_powers)

    # ── LRA : percentile 95-10 des puissances a court terme ──
    lra = 0.0
    if len(lra_powers) >= 4:
        lp = np.array(lra_powers)
        abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
        lp_f = lp[lp > abs_gate]
        if len(lp_f) >= 4:
            lra_db = -0.691 + 10.0 * np.log10(lp_f + 1e-30)
            lra = round(float(np.percentile(lra_db, 95) - np.percentile(lra_db, 10)), 2)

    # ── Mid/Side depuis accumulateurs ──
    if n_ms > 0 and n_channels >= 2:
        mid_rms = float(np.sqrt(sum_m2 / n_ms))
        side_rms = float(np.sqrt(sum_s2 / n_ms))
        ms_ratio_db = float(20 * np.log10((side_rms + 1e-12) / (mid_rms + 1e-12)))
        num_p = n_ms * sum_lr - sum_l * sum_r
        den_p = np.sqrt(abs(n_ms * sum_l2 - sum_l ** 2) * abs(n_ms * sum_r2 - sum_r ** 2))
        correlation = float(num_p / (den_p + 1e-12))
        correlation = max(-1.0, min(1.0, correlation))
        ms_dict: dict = {
            "is_stereo": True,
            "mid_rms": round(mid_rms, 6),
            "side_rms": round(side_rms, 6),
            "ms_ratio_db": round(ms_ratio_db, 2),
            "correlation": round(correlation, 4),
            "stereo_width": round(1.0 - abs(correlation), 4),
        }
    else:
        ms_dict = {
            "is_stereo": False,
            "mid_rms": 0.0, "side_rms": 0.0,
            "ms_ratio_db": 0.0, "correlation": 1.0, "stereo_width": 0.0,
        }

    return y, target_sr, rms, peak, lufs, lra, ms_dict


def _analyse_spectral_chunked(y: NDArray[Any], sr: float,
                               n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                               chunk_sec: float = 60.0) -> dict:
    """
    Calcule toutes les features spectrales par blocs de 60s.
    Pic RAM par bloc : ~50 MB au lieu de 2 GB pour la STFT complete.
    Accumule l'energie par frequence et calcule les stats globales.
    """
    chunk_size = int(chunk_sec * sr)
    n_bins = n_fft // 2 + 1
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Accumulateurs float64 (vecteurs 1D de n_bins elements, ~8 KB chacun)
    energy_sum = np.zeros(n_bins, dtype=np.float64)
    energy_sq_sum = np.zeros(n_bins, dtype=np.float64)
    log_energy_sum = np.zeros(n_bins, dtype=np.float64)
    n_frames_total = 0
    pitch_counts = np.zeros(n_bins, dtype=np.float64)

    for start in range(0, len(y), chunk_size):
        y_seg = y[start:min(start + chunk_size, len(y))]
        # STFT : pour float32 input → complex64 (8 bytes/elem)
        # Pic par bloc de 60s : 1025 * 2584 * 8 = 21 MB
        S_complex = librosa.stft(y_seg, n_fft=n_fft, hop_length=hop_length)
        S = np.abs(S_complex).astype(np.float32)
        del S_complex
        gc.collect()

        energy_sum += np.sum(S, axis=1).astype(np.float64)
        energy_sq_sum += np.sum(S.astype(np.float64) ** 2, axis=1)
        log_energy_sum += np.sum(np.log(S.astype(np.float64) + 1e-12), axis=1)
        n_frames_total += S.shape[1]

        # Pitch tracking sur ce bloc
        pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
        med_mag = float(np.median(magnitudes[magnitudes > 0])) if magnitudes.any() else 0.0
        dom_p = pitches[magnitudes > med_mag]
        for p in dom_p.ravel():
            if 0.0 < p < sr / 2:
                b = int(p * n_fft / sr)
                if 0 <= b < n_bins:
                    pitch_counts[b] += 1.0
        del S, pitches, magnitudes
        gc.collect()

    if n_frames_total == 0:
        return {}

    energy_mean = energy_sum / n_frames_total

    # Spectral centroid
    centroid = float(np.sum(frequencies * energy_mean) / (np.sum(energy_mean) + 1e-12))

    # Spectral rolloff 95%
    cum = np.cumsum(energy_mean)
    ridx = np.searchsorted(cum, 0.95 * cum[-1])
    rolloff = float(frequencies[min(ridx, n_bins - 1)])

    # Spectral flatness : exp(mean(log|S|)) / mean(|S|)
    geom = np.exp(log_energy_sum / n_frames_total)
    flatness = float(np.mean(geom / (energy_mean + 1e-12)))

    # Spectral contrast (7 sous-bandes, depuis le spectre moyen)
    sub_bands = [(0, 200), (200, 400), (400, 800), (800, 1600),
                 (1600, 3200), (3200, 6400), (6400, sr / 2)]
    contrast_mean = []
    for fmin, fmax in sub_bands:
        idx = np.where((frequencies >= fmin) & (frequencies < fmax))[0]
        if len(idx) > 5:
            be = energy_mean[idx]
            n_top = max(1, len(idx) // 5)
            peak_e = np.mean(np.sort(be)[-n_top:])
            valley_e = np.mean(np.sort(be)[:n_top])
            contrast_mean.append(round(float(np.log(peak_e / (valley_e + 1e-12) + 1e-12)), 2))
        else:
            contrast_mean.append(0.0)

    # Pitch dominant depuis histogramme
    mean_pitch = 0.0
    if pitch_counts.sum() > 0:
        peak_bin = int(np.argmax(pitch_counts))
        mean_pitch = float(peak_bin * sr / n_fft)

    # Energie par bande
    bands = {"grave": (20, 200), "bas_medium": (200, 500),
             "medium": (500, 2000), "haut_medium": (2000, 5000), "aigu": (5000, 12000)}
    band_energy: dict = {}
    band_energy_db: dict = {}
    for name, (fmin, fmax) in bands.items():
        idx = np.where((frequencies >= fmin) & (frequencies < fmax))[0]
        val = float(energy_sq_sum[idx].sum() / n_frames_total) if len(idx) > 0 else 0.0
        band_energy[name] = round(val, 4)
        band_energy_db[name] = round(10 * np.log10(val + 1e-12), 1)

    # Tessiture utile
    seuil = 0.01 * np.max(energy_mean)
    iu = np.where(energy_mean > seuil)[0]
    min_freq_utile = float(frequencies[iu[0]]) if len(iu) > 0 else 0.0
    max_freq_utile = float(frequencies[iu[-1]]) if len(iu) > 0 else 0.0

    return {
        "spectral": {
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_rolloff_hz": round(rolloff, 1),
            "spectral_flatness": round(flatness, 6),
            "spectral_contrast_db": contrast_mean,
        },
        "band_energy": band_energy,
        "band_energy_db": band_energy_db,
        "tessiture": {"min_freq": round(min_freq_utile, 1), "max_freq": round(max_freq_utile, 1)},
        "pitch_dominant": round(mean_pitch, 1),
    }


def compute_spectral_features(S: NDArray[Any], sr: int | float) -> dict:
    """
    Calcul des features spectrales a partir de la STFT partagee.
    Tout est calcule depuis S pour eviter recalcul.
    """
    # Spectral centroid (moyenne globale)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    centroid_mean = float(np.mean(centroid))

    # Spectral rolloff (95%)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)
    rolloff_mean = float(np.mean(rolloff))

    # Spectral flatness (0 = tonal, 1 = bruit)
    flatness = librosa.feature.spectral_flatness(S=S)
    flatness_mean = float(np.mean(flatness))

    # Spectral contrast (7 bandes par defaut)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    contrast_mean = np.mean(contrast, axis=1).tolist()
    contrast_mean = [round(c, 2) for c in contrast_mean]

    return {
        "spectral_centroid_hz": round(centroid_mean, 1),
        "spectral_rolloff_hz": round(rolloff_mean, 1),
        "spectral_flatness": round(flatness_mean, 6),
        "spectral_contrast_db": contrast_mean,
    }


def generate_interpretation(analyse: dict) -> dict:
    """
    Genere des conseils automatiques de mixage selon :
    - Crest factor
    - LUFS
    - Spectral centroid
    - MS ratio
    """
    conseils = []
    crest = analyse.get("crest_factor", 0)
    lufs = analyse.get("lufs", -99)
    centroid = analyse.get("spectral", {}).get("spectral_centroid_hz", 0)
    ms = analyse.get("mid_side", {})
    ms_ratio = ms.get("ms_ratio_db", 0)

    # ── Crest factor ──
    if crest < 6:
        conseils.append({
            "parametre": "crest_factor",
            "valeur": round(crest, 2),
            "diagnostic": "Tres compresse / ecrase",
            "conseil": "Reduire la compression ou le limiting. Le mix manque de dynamique.",
        })
    elif crest < 10:
        conseils.append({
            "parametre": "crest_factor",
            "valeur": round(crest, 2),
            "diagnostic": "Compression moderee",
            "conseil": "Bon niveau de compression pour du pop/rock/EDM.",
        })
    elif crest < 18:
        conseils.append({
            "parametre": "crest_factor",
            "valeur": round(crest, 2),
            "diagnostic": "Bonne dynamique",
            "conseil": "Dynamique naturelle, ideal pour jazz/classique/acoustique.",
        })
    else:
        conseils.append({
            "parametre": "crest_factor",
            "valeur": round(crest, 2),
            "diagnostic": "Tres dynamique",
            "conseil": "Verifier si le mix n'est pas trop faible en loudness.",
        })

    # ── LUFS ──
    if lufs > -8:
        conseils.append({
            "parametre": "lufs",
            "valeur": round(lufs, 2),
            "diagnostic": "Mix tres loud (loudness war)",
            "conseil": "Baisser le limiter. Les plateformes vont reduire le gain.",
        })
    elif lufs > -14:
        conseils.append({
            "parametre": "lufs",
            "valeur": round(lufs, 2),
            "diagnostic": "Bon pour le streaming",
            "conseil": "Niveau optimal pour Spotify/YouTube (-14 LUFS).",
        })
    elif lufs > -20:
        conseils.append({
            "parametre": "lufs",
            "valeur": round(lufs, 2),
            "diagnostic": "Mix tranquille",
            "conseil": "Acceptable pour du contenu dynamique (classique, podcast narratif).",
        })
    else:
        conseils.append({
            "parametre": "lufs",
            "valeur": round(lufs, 2),
            "diagnostic": "Mix tres bas",
            "conseil": "Ajouter du gain ou verifier la chaine de mastering.",
        })

    # ── Spectral centroid ──
    if centroid > 0:
        if centroid > 4000:
            conseils.append({
                "parametre": "spectral_centroid",
                "valeur": round(centroid, 1),
                "diagnostic": "Mix tres brillant",
                "conseil": "Verifier agressivite dans les aigus. Potentiel fatigue auditive.",
            })
        elif centroid > 2500:
            conseils.append({
                "parametre": "spectral_centroid",
                "valeur": round(centroid, 1),
                "diagnostic": "Equilibre brillant",
                "conseil": "Balance frequentielle orientee presence/clarte.",
            })
        elif centroid > 1200:
            conseils.append({
                "parametre": "spectral_centroid",
                "valeur": round(centroid, 1),
                "diagnostic": "Balance neutre",
                "conseil": "Centroid dans la zone mediane, equilibre naturel.",
            })
        else:
            conseils.append({
                "parametre": "spectral_centroid",
                "valeur": round(centroid, 1),
                "diagnostic": "Mix sombre",
                "conseil": "Manque potentiel de clarte. Envisager un boost en presence (2-5 kHz).",
            })

    # ── MS ratio ──
    if ms.get("is_stereo", False):
        if ms_ratio > -3:
            conseils.append({
                "parametre": "ms_ratio",
                "valeur": round(ms_ratio, 2),
                "diagnostic": "Side tres present",
                "conseil": "Image stereo large, verifier la compatibilite mono.",
            })
        elif ms_ratio > -10:
            conseils.append({
                "parametre": "ms_ratio",
                "valeur": round(ms_ratio, 2),
                "diagnostic": "Balance M/S correcte",
                "conseil": "Bon ratio Mid/Side pour un mix stereo standard.",
            })
        else:
            conseils.append({
                "parametre": "ms_ratio",
                "valeur": round(ms_ratio, 2),
                "diagnostic": "Mix tres centre (quasi mono)",
                "conseil": "Peu de contenu lateral. Elargir avec reverb/chorus/panning si necessaire.",
            })

    return {"conseils_mixage": conseils}


def compare_reference(track: str, reference: str) -> dict:
    """
    Compare deux fichiers audio et genere un rapport detaille :
    LUFS, Crest Factor, Centroid, MS ratio, True Peak.
    """
    def _extract_metrics(fichier: str) -> dict:
        # Streaming : LUFS/LRA/M/S calcules a la volee, jamais le fichier entier en RAM
        y, sr, rms, peak, lufs, lra, ms = _stream_load_mono(fichier, target_sr=22050)

        # Crest factor
        crest = 20 * np.log10((peak + 1e-9) / (rms + 1e-9))
        true_peak_db = round(20 * np.log10(peak + 1e-12), 2)

        # STFT unique pour spectral features
        S, _ = compute_stft(y, sr)
        del y
        gc.collect()
        centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))

        return {
            "fichier": fichier,
            "lufs": round(lufs, 2),
            "crest_factor": round(float(crest), 2),
            "true_peak_db": true_peak_db,
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_rolloff_hz": round(rolloff, 1),
            "spectral_flatness": round(flatness, 6),
            "lra_lu": lra,
            "ms_ratio_db": ms["ms_ratio_db"],
            "stereo_width": ms["stereo_width"],
            "correlation": ms["correlation"],
        }

    console.print(f"[cyan]Extraction metriques : {os.path.basename(track)}[/cyan]")
    m_track = _extract_metrics(track)
    console.print(f"[cyan]Extraction metriques : {os.path.basename(reference)}[/cyan]")
    m_ref = _extract_metrics(reference)

    # Calcul des differences
    diff_keys = [
        "lufs", "crest_factor", "true_peak_db", "spectral_centroid_hz",
        "spectral_rolloff_hz", "spectral_flatness", "lra_lu",
        "ms_ratio_db", "stereo_width", "correlation",
    ]
    differences = {}
    for k in diff_keys:
        val_t = m_track.get(k, 0)
        val_r = m_ref.get(k, 0)
        differences[k] = round(val_t - val_r, 4)

    # Conseils de rapprochement
    conseils = []
    if abs(differences["lufs"]) > 2:
        direction = "baisser" if differences["lufs"] > 0 else "augmenter"
        conseils.append(f"LUFS : {direction} le loudness de {abs(differences['lufs']):.1f} LU")

    if abs(differences["crest_factor"]) > 3:
        if differences["crest_factor"] > 0:
            conseils.append("Crest factor : votre mix est plus dynamique que la ref. Compresser davantage.")
        else:
            conseils.append("Crest factor : votre mix est plus compresse que la ref. Allegez la compression.")

    if abs(differences["spectral_centroid_hz"]) > 500:
        if differences["spectral_centroid_hz"] > 0:
            conseils.append("Centroid : votre mix est plus brillant. Reduire les aigus ou booster les graves.")
        else:
            conseils.append("Centroid : votre mix est plus sombre. Booster la presence/air.")

    if abs(differences["true_peak_db"]) > 1:
        conseils.append(f"True Peak : ecart de {abs(differences['true_peak_db']):.1f} dB. Ajuster le limiter.")

    if abs(differences["ms_ratio_db"]) > 3:
        if differences["ms_ratio_db"] > 0:
            conseils.append("M/S : votre mix est plus large que la ref. Reduire le side.")
        else:
            conseils.append("M/S : votre mix est plus etroit. Elargir l'image stereo.")

    rapport = {
        "track": m_track,
        "reference": m_ref,
        "differences": differences,
        "conseils_rapprochement": conseils,
    }

    # Affichage
    _print_comparison(rapport)

    # Export JSON
    json_name = "compare_reference.json"
    with open(json_name, "w") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    console.print(f"\n[green]Rapport sauvegarde : {json_name}[/green]")

    return rapport


def _print_comparison(rapport: dict) -> None:
    """Affichage du rapport de comparaison."""
    m_track = rapport["track"]
    m_ref = rapport["reference"]
    diff = rapport["differences"]

    table = Table(title="Comparaison Track vs Reference")
    table.add_column("Metrique", style="cyan")
    table.add_column("Track", justify="right")
    table.add_column("Reference", justify="right")
    table.add_column("Delta", justify="right")

    rows = [
        ("LUFS", "lufs", ""),
        ("Crest Factor (dB)", "crest_factor", ""),
        ("True Peak (dBTP)", "true_peak_db", ""),
        ("Centroid (Hz)", "spectral_centroid_hz", ""),
        ("Rolloff (Hz)", "spectral_rolloff_hz", ""),
        ("Flatness", "spectral_flatness", ""),
        ("LRA (LU)", "lra_lu", ""),
        ("M/S Ratio (dB)", "ms_ratio_db", ""),
        ("Stereo Width", "stereo_width", ""),
        ("Correlation", "correlation", ""),
    ]

    for label, key, _ in rows:
        vt = m_track.get(key, 0)
        vr = m_ref.get(key, 0)
        d = diff.get(key, 0)
        color = "green" if abs(d) < 1 else ("yellow" if abs(d) < 3 else "red")
        table.add_row(label, f"{vt}", f"{vr}", f"[{color}]{d:+.2f}[/{color}]")

    console.print(table)

    conseils = rapport.get("conseils_rapprochement", [])
    if conseils:
        console.print("\n[bold magenta]Conseils de rapprochement :[/bold magenta]")
        for c in conseils:
            console.print(f"  [yellow]> {c}[/yellow]")


def analyse_audio(fichier: str, plot_spectre: bool = False) -> None:
    """
    Analyse audio complete orientee mixage/mastering.
    Chargement par blocs de 30s (streaming) : pic memoire ~15 MB par bloc.
    Permet d'analyser des fichiers de plusieurs heures sur machine limitee.
    """
    # ── Phase 1 : streaming → y mono 22050 Hz + stats pre-calculees ──
    # Pic RAM pendant le streaming : ~15 MB par bloc
    # LUFS, LRA et Mid/Side calcules pendant le streaming (pas de gros tableau float64)
    y, sr, rms, peak, lufs, lra, ms_analysis = _stream_load_mono(fichier, target_sr=22050)
    # y : ~320 MB pour 60 min (float32, mono, 22050 Hz)

    duree_sec = len(y) / sr
    true_peak_db = round(20 * np.log10(peak + 1e-12), 2)
    crest_factor = 20 * np.log10((peak + 1e-9) / (rms + 1e-9))

    # ── Segment representatif 5 min (vue de y, 0 copie) ──
    repr_samples = min(len(y), int(5 * 60 * sr))
    y_repr = y[:repr_samples]

    # ── SNR : sous-echantillonnage 1/100 pour eviter copie abs de 318 MB ──
    noise_floor = float(np.percentile(np.abs(y[::100]), 5))
    snr = 20 * np.log10(peak / (noise_floor + 1e-9))

    # ── Dynamique RMS sur segment 5 min (matrice frames ~212 MB, pas 635 MB) ──
    frame_length = int(0.05 * sr)
    hop_rms = int(0.025 * sr)
    rms_frames = librosa.feature.rms(y=y_repr, frame_length=frame_length, hop_length=hop_rms)[0]
    dynamique_rms = float(np.std(rms_frames))
    del rms_frames

    # Tempo (STFT interne sur 5 min ~100 MB)
    tempo, _ = librosa.beat.beat_track(y=y_repr, sr=sr)

    # Tonalite (CQT interne)
    key = detect_key(y_repr, sr)

    # Transitoires (STFT interne sur 5 min ~100 MB)
    onset_env = librosa.onset.onset_strength(y=y_repr, sr=sr)
    transients = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    nb_transitoires = len(transients)
    del onset_env, y_repr
    gc.collect()

    # ── STFT chunked : blocs de 60s, pic ~50 MB/bloc au lieu de 2 GB ──
    sp_results = _analyse_spectral_chunked(y, sr)
    del y
    gc.collect()

    spectral = sp_results.get("spectral", {})
    band_energy = sp_results.get("band_energy", {})
    band_energy_db = sp_results.get("band_energy_db", {})
    tessiture = sp_results.get("tessiture", {"min_freq": 0.0, "max_freq": 0.0})
    min_freq_utile = tessiture["min_freq"]
    max_freq_utile = tessiture["max_freq"]
    mean_pitch = sp_results.get("pitch_dominant", 0.0)

    # ── Construction du resultat ──
    analyse = {
        "fichier": fichier,
        "duree_sec": round(duree_sec, 2),
        "tempo": float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "lufs": round(lufs, 2),
        "true_peak_db": true_peak_db,
        "lra_lu": lra,
        "snr_db": round(float(snr), 2),
        "tessiture_utile": {
            "min_freq": round(min_freq_utile, 1),
            "max_freq": round(max_freq_utile, 1),
        },
        "pitch_dominant": round(mean_pitch, 1),
        "key": key,
        "spectral": spectral,
        "energie_par_bande": band_energy,
        "energie_par_bande_db": band_energy_db,
        "mid_side": ms_analysis,
        "nb_transitoires": nb_transitoires,
        "crest_factor": round(float(crest_factor), 2),
        "dynamique_rms": round(dynamique_rms, 6),
    }

    # ── Interpretation et conseils mixage ──
    interpretation = generate_interpretation(analyse)
    analyse["analyse_interpretation"] = interpretation

    # ── Export JSON ──
    json_name = os.path.splitext(os.path.basename(fichier))[0] + ".json"
    with open(json_name, "w") as f:
        json.dump(analyse, f, indent=2, ensure_ascii=False)

    # ── Export CSV ──
    csv_name = "analyse_audio.csv"
    write_header = not os.path.exists(csv_name)
    with open(csv_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "fichier", "duree_sec", "tempo", "rms", "peak", "lufs",
                "true_peak_db", "lra_lu", "snr_db",
                "min_freq", "max_freq", "pitch_dominant", "key",
                "spectral_centroid", "spectral_rolloff", "spectral_flatness",
                "stereo_width", "ms_ratio_db", "correlation",
                "nb_transitoires", "crest_factor", "dynamique_rms",
                "grave", "bas_medium", "medium", "haut_medium", "aigu",
            ])
        writer.writerow([
            fichier, duree_sec, analyse["tempo"], rms, peak, lufs,
            true_peak_db, lra, snr,
            min_freq_utile, max_freq_utile, mean_pitch, key,
            spectral["spectral_centroid_hz"], spectral["spectral_rolloff_hz"],
            spectral["spectral_flatness"],
            ms_analysis["stereo_width"], ms_analysis["ms_ratio_db"],
            ms_analysis["correlation"],
            nb_transitoires, crest_factor, dynamique_rms,
            band_energy["grave"], band_energy["bas_medium"],
            band_energy["medium"], band_energy["haut_medium"],
            band_energy["aigu"],
        ])

    # ── Affichage console ──
    _print_analysis(analyse, json_name, csv_name)

    # ── Graphique ──
    if plot_spectre:
        plt.figure(figsize=(10, 4))
        plt.semilogx(frequencies[1:], energy[1:])
        plt.title(f"Spectre d'energie - {os.path.basename(fichier)}")
        plt.xlabel("Frequence (Hz)")
        plt.ylabel("Energie")
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.show()


def _print_analysis(analyse: dict, json_name: str, csv_name: str) -> None:
    """Affichage synthetique enrichi."""
    ms = analyse.get("mid_side", {})
    sp = analyse.get("spectral", {})

    console.print(Panel.fit(
        f"[bold cyan]Analyse pro : {analyse['fichier']}[/bold cyan]\n"
        f"Duree : [bold]{int(analyse['duree_sec'] // 60)} min {int(analyse['duree_sec'] % 60)} sec[/bold]\n"
        f"TEMPO : [bold]{analyse['tempo']:.1f} BPM[/bold]\n"
        f"RMS : [bold]{analyse['rms']:.4f}[/bold] | Peak : [bold]{analyse['peak']:.4f}[/bold]\n"
        f"LUFS : [bold]{analyse['lufs']:.2f}[/bold] | True Peak : [bold]{analyse['true_peak_db']:.2f} dBTP[/bold]\n"
        f"LRA : [bold]{analyse['lra_lu']:.2f} LU[/bold] | SNR : [bold]{analyse['snr_db']:.2f} dB[/bold]\n"
        f"Tessiture : [bold]{analyse['tessiture_utile']['min_freq']:.1f}[/bold] - "
        f"[bold]{analyse['tessiture_utile']['max_freq']:.1f} Hz[/bold]\n"
        f"Pitch dominant : [bold]{analyse['pitch_dominant']:.1f} Hz[/bold] | "
        f"Tonalite : [bold]{analyse['key']}[/bold]\n"
        f"Centroid : [bold]{sp.get('spectral_centroid_hz', 0):.1f} Hz[/bold] | "
        f"Rolloff 95% : [bold]{sp.get('spectral_rolloff_hz', 0):.1f} Hz[/bold]\n"
        f"Flatness : [bold]{sp.get('spectral_flatness', 0):.6f}[/bold]\n"
        f"Crest factor : [bold]{analyse['crest_factor']:.2f} dB[/bold] | "
        f"Dynamique RMS : [bold]{analyse['dynamique_rms']:.4f}[/bold]\n"
        f"M/S Ratio : [bold]{ms.get('ms_ratio_db', 0):.2f} dB[/bold] | "
        f"Stereo Width : [bold]{ms.get('stereo_width', 0):.4f}[/bold] | "
        f"Corr : [bold]{ms.get('correlation', 0):.4f}[/bold]\n"
        f"Transitoires : [bold]{analyse['nb_transitoires']}[/bold]\n"
        f"Energie (dB) : {analyse.get('energie_par_bande_db', {})}\n"
        f"[green]JSON : {json_name} | CSV : {csv_name}[/green]"
    ))

    # Conseils mixage
    interp = analyse.get("analyse_interpretation", {})
    conseils = interp.get("conseils_mixage", [])
    if conseils:
        table = Table(title="Interpretation & Conseils Mixage")
        table.add_column("Parametre", style="cyan")
        table.add_column("Valeur", justify="right")
        table.add_column("Diagnostic", style="yellow")
        table.add_column("Conseil")
        for c in conseils:
            table.add_row(
                c["parametre"], str(c["valeur"]),
                c["diagnostic"], c["conseil"],
            )
        console.print(table)


def batch_analyse(folder: str, plot_spectre: bool = False) -> None:
    """Analyse batch de tous les fichiers audio d'un dossier."""
    audio_exts = (".wav", ".mp3", ".flac", ".aiff", ".ogg")
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(audio_exts)]
    if not files:
        console.print(f"[yellow]Aucun fichier audio dans {folder}[/yellow]")
        return
    for f in files:
        console.print(f"\n[cyan]--- {os.path.basename(f)} ---[/cyan]")
        try:
            analyse_audio(f, plot_spectre=plot_spectre)
        except Exception as e:
            console.print(f"[red]Erreur sur {f} : {e}[/red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse audio pro (mixage/mastering, LUFS, True Peak, M/S, LRA, spectral, compare ref)",
        epilog="Exemples:\n"
               "  python audio_analyse.py mon_mix.wav\n"
               "  python audio_analyse.py -f mon_mix.wav --plot\n"
               "  python audio_analyse.py -d ./compo/audio\n"
               "  python audio_analyse.py --compare mon_mix.wav --ref reference.wav\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fichier", nargs="?", help="Fichier audio a analyser (sans option)")
    parser.add_argument("-f", "--file", help="Fichier audio a analyser (avec option)")
    parser.add_argument("-d", "--dir", help="Dossier a analyser en mode batch")
    parser.add_argument("--plot", action="store_true", help="Afficher le graphique du spectre")
    parser.add_argument("--compare", help="Fichier track a comparer")
    parser.add_argument("--ref", help="Fichier reference pour la comparaison")
    args = parser.parse_args()

    # Mode comparaison
    if args.compare and args.ref:
        if not os.path.exists(args.compare):
            console.print(f"[red]Fichier introuvable : {args.compare}[/red]")
        elif not os.path.exists(args.ref):
            console.print(f"[red]Fichier introuvable : {args.ref}[/red]")
        else:
            console.print(f"[cyan]Comparaison : {args.compare} vs {args.ref}[/cyan]\n")
            compare_reference(args.compare, args.ref)
    elif args.compare or args.ref:
        console.print("[red]--compare et --ref doivent etre utilises ensemble.[/red]")
    else:
        # Mode analyse standard
        fichier_cible = args.fichier or args.file

        if fichier_cible:
            if os.path.exists(fichier_cible):
                console.print(f"[cyan]Analyse de : {fichier_cible}[/cyan]\n")
                analyse_audio(fichier_cible, plot_spectre=args.plot)
            else:
                console.print(f"[red]Fichier introuvable : {fichier_cible}[/red]")
        elif args.dir:
            if os.path.exists(args.dir):
                console.print(f"[cyan]Analyse du dossier : {args.dir}[/cyan]\n")
                batch_analyse(args.dir, plot_spectre=args.plot)
            else:
                console.print(f"[red]Dossier introuvable : {args.dir}[/red]")
        else:
            console.print("[yellow]Utilisation :[/yellow]")
            console.print("  [green]python audio_analyse.py mon_mix.wav[/green]")
            console.print("  [green]python audio_analyse.py -f mon_mix.wav --plot[/green]")
            console.print("  [green]python audio_analyse.py -d ./compo/audio[/green]")
            console.print("  [green]python audio_analyse.py --compare mix.wav --ref ref.wav[/green]")
            console.print("\n[dim]--help pour plus d'informations[/dim]")
