#!/usr/bin/env python3
"""
Analyse de piste individuelle (kick, snare, voix, basse, synthe)
- Detection de resonances etroites (pics locaux significatifs)
- Bande contenant 90% de l'energie
- Zones critiques : Mud, Boxiness, Harshness, Air
- Recommandations EQ automatiques (frequence, type, gain, Q)
- STFT optimise (calcul unique)
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
from scipy.signal import find_peaks
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

# Zones critiques (Hz)
CRITICAL_ZONES = {
    "mud":       (200, 400),
    "boxiness":  (400, 800),
    "harshness": (2000, 5000),
    "air":       (8000, 15000),
}

# Bandes d'energie standard
ENERGY_BANDS = {
    "sub":        (20, 60),
    "grave":      (60, 200),
    "bas_medium": (200, 500),
    "medium":     (500, 2000),
    "haut_medium":(2000, 5000),
    "aigu":       (5000, 12000),
    "brillance":  (12000, 20000),
}

# Seuils de detection par zone (dB au-dessus de la moyenne locale)
ZONE_THRESHOLDS_DB = {
    "mud":       3.0,
    "boxiness":  3.0,
    "harshness": 4.0,
    "air":      -2.0,  # seuil negatif = on detecte le manque
}


def detect_key(y: NDArray[Any], sr: int | float) -> str:
    """Detection de la tonalite via chroma CQT."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return notes[np.argmax(chroma_mean)]
    except Exception:
        return "Indeterminee"


def compute_stft(y: NDArray[Any], n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> tuple:
    """Calcul unique de la STFT. Retourne (S_magnitude, S_db, frequencies)."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    frequencies = librosa.fft_frequencies(sr=22050, n_fft=n_fft)
    return S, S_db, frequencies


def compute_stft_full(y: NDArray[Any], sr: int | float, n_fft: int = N_FFT,
                      hop_length: int = HOP_LENGTH) -> tuple:
    """STFT avec frequences reelles basees sur le sr. S_db non calcule (economie memoire)."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S, None, frequencies


def detect_resonances(S: NDArray[Any], frequencies: NDArray[Any],
                      prominence_db: float = 6.0, min_freq: float = 30.0,
                      max_freq: float = 18000.0) -> list[dict]:
    """
    Detection des resonances etroites (pics locaux significatifs).
    Travaille sur le spectre moyen en dB.
    """
    # Spectre moyen en dB
    mean_spectrum = np.mean(S, axis=1)
    mean_spectrum_db = 20 * np.log10(mean_spectrum + 1e-12)

    # Filtrer la plage de frequences
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    valid_indices = np.where(freq_mask)[0]

    if len(valid_indices) == 0:
        return []

    spectrum_slice = mean_spectrum_db[valid_indices]

    # Detection de pics avec scipy
    peaks, properties = find_peaks(
        spectrum_slice,
        prominence=prominence_db,
        distance=3,  # distance minimale entre pics (bins)
        width=(1, 15),  # largeur en bins : resonances etroites
    )

    resonances = []
    for i, peak_idx in enumerate(peaks):
        actual_idx = valid_indices[peak_idx]
        freq = float(frequencies[actual_idx])
        level_db = float(mean_spectrum_db[actual_idx])
        prominence = float(properties["prominences"][i])

        # Estimer le Q a partir de la largeur du pic
        width_bins = float(properties["widths"][i])
        freq_resolution = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 1.0
        bandwidth_hz = width_bins * freq_resolution
        q_estimated = freq / (bandwidth_hz + 1e-6)

        resonances.append({
            "frequence_hz": round(freq, 1),
            "niveau_db": round(level_db, 1),
            "prominence_db": round(prominence, 1),
            "q_estime": round(min(q_estimated, 30.0), 1),
            "bandwidth_hz": round(bandwidth_hz, 1),
        })

    # Trier par prominence decroissante
    resonances.sort(key=lambda r: r["prominence_db"], reverse=True)
    return resonances[:15]  # Top 15 resonances


def compute_energy_band_90(S: NDArray[Any], frequencies: NDArray[Any]) -> dict:
    """
    Calcul de la bande contenant 90% de l'energie totale.
    Retourne les bornes basse et haute.
    """
    energy = np.sum(S ** 2, axis=1)
    total_energy = np.sum(energy)

    if total_energy < 1e-12:
        return {"freq_low_hz": 0.0, "freq_high_hz": 0.0, "energy_ratio": 0.0}

    cumulative = np.cumsum(energy)
    cumulative_norm = cumulative / total_energy

    # 5% et 95% pour centrer la bande a 90%
    idx_low = np.searchsorted(cumulative_norm, 0.05)
    idx_high = np.searchsorted(cumulative_norm, 0.95)

    idx_low = max(0, min(idx_low, len(frequencies) - 1))
    idx_high = max(0, min(idx_high, len(frequencies) - 1))

    return {
        "freq_low_hz": round(float(frequencies[idx_low]), 1),
        "freq_high_hz": round(float(frequencies[idx_high]), 1),
        "energy_ratio": 0.90,
    }


def analyse_critical_zones(S: NDArray[Any], frequencies: NDArray[Any]) -> dict:
    """
    Analyse des zones critiques : Mud, Boxiness, Harshness, Air.
    Compare l'energie de chaque zone a la moyenne globale.
    """
    energy = np.sum(S ** 2, axis=1)
    total_energy = np.sum(energy)
    if total_energy < 1e-12:
        return {}

    # Moyenne d'energie par bin (reference globale)
    mean_energy_per_bin = total_energy / len(energy)

    zones_analysis = {}
    for zone_name, (fmin, fmax) in CRITICAL_ZONES.items():
        idx = np.where((frequencies >= fmin) & (frequencies < fmax))[0]
        if len(idx) == 0:
            zones_analysis[zone_name] = {
                "energy_db": -120.0,
                "relative_db": 0.0,
                "problematic": False,
            }
            continue

        zone_energy = np.sum(energy[idx])
        zone_mean_per_bin = zone_energy / len(idx)

        # En dB relatif a la moyenne globale
        relative_db = 10 * np.log10((zone_mean_per_bin + 1e-12) / (mean_energy_per_bin + 1e-12))
        energy_db = 10 * np.log10(zone_energy + 1e-12)

        threshold = ZONE_THRESHOLDS_DB[zone_name]
        if zone_name == "air":
            # Pour "air", on detecte le manque (trop peu d'energie)
            problematic = relative_db < threshold
        else:
            problematic = relative_db > threshold

        zones_analysis[zone_name] = {
            "energy_db": round(float(energy_db), 1),
            "relative_db": round(float(relative_db), 1),
            "problematic": bool(problematic),
        }

    return zones_analysis


def generate_eq_recommendations(resonances: list[dict],
                                zones_analysis: dict) -> list[dict]:
    """
    Genere des recommandations EQ basees sur les resonances et zones critiques.
    """
    recommendations = []

    # 1. Couper les resonances excessives
    for res in resonances[:8]:
        if res["prominence_db"] > 6.0:
            gain = -min(res["prominence_db"] * 0.5, 6.0)
            recommendations.append({
                "frequence_hz": res["frequence_hz"],
                "type": "cut",
                "gain_db": round(gain, 1),
                "q_suggere": round(min(res["q_estime"], 12.0), 1),
                "raison": f"Resonance etroite ({res['prominence_db']:.1f} dB prominence)",
            })

    # 2. Traiter les zones critiques problematiques
    zone_actions = {
        "mud":       {"type": "cut", "freq_center": 300,   "q": 1.5, "max_cut": -4.0,
                      "raison": "Accumulation Mud (200-400 Hz)"},
        "boxiness":  {"type": "cut", "freq_center": 600,   "q": 2.0, "max_cut": -3.0,
                      "raison": "Boxiness (400-800 Hz)"},
        "harshness": {"type": "cut", "freq_center": 3500,  "q": 2.0, "max_cut": -3.0,
                      "raison": "Harshness (2-5 kHz)"},
        "air":       {"type": "boost", "freq_center": 12000, "q": 0.7, "max_boost": 3.0,
                      "raison": "Manque d'air (8-15 kHz)"},
    }

    for zone_name, info in zones_analysis.items():
        if not info.get("problematic", False):
            continue

        action = zone_actions.get(zone_name)
        if action is None:
            continue

        if action["type"] == "cut":
            gain = max(-abs(info["relative_db"]) * 0.4, action["max_cut"])
        else:
            gain = min(abs(info["relative_db"]) * 0.3, action.get("max_boost", 3.0))

        recommendations.append({
            "frequence_hz": action["freq_center"],
            "type": action["type"],
            "gain_db": round(gain, 1),
            "q_suggere": action["q"],
            "raison": action["raison"],
        })

    return recommendations


def find_audio_start(fichier: str, threshold: float = 0.001, block_sec: int = 10) -> float:
    """
    Scanne le fichier par blocs pour trouver le debut reel de l'audio.
    Retourne l'offset en secondes (0.0 si l'audio commence immediatement).
    Utile pour les bounces DAW qui demarrent au t=0 de la timeline.
    """
    with sf.SoundFile(fichier) as f:
        block_size = f.samplerate * block_sec
        for i, block in enumerate(f.blocks(blocksize=block_size, dtype='float32')):
            if np.max(np.abs(block)) > threshold:
                offset_sec = i * block_sec
                if offset_sec > 0:
                    console.print(f"[dim]Silence initial detecte : debut audio a {offset_sec} s[/dim]")
                return float(offset_sec)
    return 0.0


def analyse_piste(fichier: str, plot_spectre: bool = False, output_dir: str = "",
                  duration: float | None = None) -> dict:
    """
    Analyse complete d'une piste individuelle.
    STFT calculee une seule fois et reutilisee partout.
    duration : limiter l'analyse aux N premieres secondes (None = fichier entier).
    """
    # ── Detection automatique du debut audio (saute le silence de timeline DAW) ──
    offset = find_audio_start(fichier)

    # ── Chargement unique en stereo (evite les rechargements) ──
    y_stereo, sr = librosa.load(fichier, mono=False, sr=22050, offset=offset, duration=duration)

    # Derive mono depuis le stereo (evite un 2eme chargement)
    if y_stereo.ndim == 2:
        y = np.mean(y_stereo, axis=0).astype(np.float32)
        left, right = y_stereo[0], y_stereo[1]
        corr = float(np.corrcoef(left, right)[0, 1])
        stereo_width = round(1.0 - abs(corr), 4)
        # LUFS : copie explicite pour liberer y_stereo apres
        try:
            data_lufs = np.asfortranarray(y_stereo.T)  # copie reelle, pas une vue
            meter = pyln.Meter(int(sr))
            lufs = float(meter.integrated_loudness(data_lufs))
            del data_lufs
        except Exception:
            lufs = -99.0
        del left, right, y_stereo
        gc.collect()
    else:
        y = y_stereo.astype(np.float32)
        del y_stereo
        stereo_width = 0.0
        try:
            data_lufs = y.reshape(-1, 1).copy()
            meter = pyln.Meter(int(sr))
            lufs = float(meter.integrated_loudness(data_lufs))
            del data_lufs
        except Exception:
            lufs = -99.0
        gc.collect()

    duree_sec = librosa.get_duration(y=y, sr=sr)

    # ── STFT unique (optimisation) ──
    S, _, frequencies = compute_stft_full(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # ── Tempo ──
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # ── RMS & Peak ──
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))

    # ── Energie par bande depuis la STFT partagee ──
    energy = np.sum(S ** 2, axis=1)
    band_energy = {}
    for name, (fmin, fmax) in ENERGY_BANDS.items():
        idx = np.where((frequencies >= fmin) & (frequencies < fmax))[0]
        band_energy[name] = round(float(np.sum(energy[idx])), 4) if len(idx) > 0 else 0.0

    # ── Energie par bande en dB ──
    band_energy_db = {}
    for name, val in band_energy.items():
        band_energy_db[name] = round(10 * np.log10(val + 1e-12), 1)

    # ── Tessiture utile ──
    seuil = 0.01 * np.max(energy)
    indices_utiles = np.where(energy > seuil)[0]
    min_freq_utile = float(frequencies[indices_utiles[0]]) if len(indices_utiles) > 0 else 0.0
    max_freq_utile = float(frequencies[indices_utiles[-1]]) if len(indices_utiles) > 0 else 0.0

    # ── Pitch dominant ──
    pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
    dominant_pitches = pitches[magnitudes > np.median(magnitudes)]
    mean_pitch = float(np.mean(dominant_pitches)) if len(dominant_pitches) > 0 else 0.0

    # ── Tonalite ──
    key = detect_key(y, sr)

    # ── SNR ──
    noise_floor = np.percentile(np.abs(y), 5)
    snr = 20 * np.log10(peak / (noise_floor + 1e-9))

    # ── Transitoires ──
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    transients = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    nb_transitoires = len(transients)

    # ── Dynamique ──
    crest_factor = 20 * np.log10((peak + 1e-9) / (rms + 1e-9))
    frame_length = int(0.05 * sr)
    hop_rms = int(0.025 * sr)
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_rms)[0]
    dynamique_rms = float(np.std(rms_frames))

    del y, onset_env, rms_frames  # liberer l'audio mono - plus besoin
    gc.collect()

    # ── Spectral features (depuis la STFT partagee) ──
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))

    # ── ANALYSES AVANCEES PISTE INDIVIDUELLE ──

    # Detection de resonances
    resonances = detect_resonances(S, frequencies, prominence_db=6.0)

    # Bande 90% energie
    energy_band_90 = compute_energy_band_90(S, frequencies)

    # Zones critiques
    zones_critiques = analyse_critical_zones(S, frequencies)

    # Recommandations EQ
    recommandations_eq = generate_eq_recommendations(resonances, zones_critiques)

    # ── Construction du resultat ──
    analyse = {
        "fichier": fichier,
        "duree_sec": round(duree_sec, 2),
        "tempo": float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "lufs": round(lufs, 2),
        "snr_db": round(float(snr), 2),
        "tessiture_utile": {
            "min_freq": round(min_freq_utile, 1),
            "max_freq": round(max_freq_utile, 1),
        },
        "pitch_dominant": round(mean_pitch, 1),
        "key": key,
        "stereo_width": round(stereo_width, 4),
        "nb_transitoires": nb_transitoires,
        "crest_factor": round(crest_factor, 2),
        "dynamique_rms": round(dynamique_rms, 6),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "spectral_rolloff_hz": round(spectral_rolloff, 1),
        "spectral_flatness": round(spectral_flatness, 6),
        "energie_par_bande": band_energy,
        "energie_par_bande_db": band_energy_db,
        "bande_90pct_energie": energy_band_90,
        "resonances_detectees": resonances,
        "zones_critiques": zones_critiques,
        "recommandations_eq": recommandations_eq,
    }

    # ── Export JSON ──
    base_name = os.path.splitext(os.path.basename(fichier))[0] + "_piste.json"
    json_name = os.path.join(output_dir, base_name) if output_dir else base_name
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(json_name, "w") as f:
        json.dump(analyse, f, indent=2, ensure_ascii=False)

    # ── Export CSV ──
    csv_name = os.path.join(output_dir, "analyse_piste.csv") if output_dir else "analyse_piste.csv"
    write_header = not os.path.exists(csv_name)
    with open(csv_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "fichier", "duree_sec", "tempo", "rms", "peak", "lufs", "snr_db",
                "min_freq", "max_freq", "pitch_dominant", "key", "stereo_width",
                "nb_transitoires", "crest_factor", "dynamique_rms",
                "spectral_centroid", "spectral_rolloff", "spectral_flatness",
                "band_90_low", "band_90_high",
                "mud_db", "mud_problematic",
                "boxiness_db", "boxiness_problematic",
                "harshness_db", "harshness_problematic",
                "air_db", "air_problematic",
                "nb_resonances", "nb_eq_recommandations",
            ])
        mud = zones_critiques.get("mud", {})
        box = zones_critiques.get("boxiness", {})
        harsh = zones_critiques.get("harshness", {})
        air = zones_critiques.get("air", {})
        writer.writerow([
            fichier, duree_sec, analyse["tempo"], rms, peak, lufs, snr,
            min_freq_utile, max_freq_utile, mean_pitch, key, stereo_width,
            nb_transitoires, crest_factor, dynamique_rms,
            spectral_centroid, spectral_rolloff, spectral_flatness,
            energy_band_90["freq_low_hz"], energy_band_90["freq_high_hz"],
            mud.get("relative_db", 0), mud.get("problematic", False),
            box.get("relative_db", 0), box.get("problematic", False),
            harsh.get("relative_db", 0), harsh.get("problematic", False),
            air.get("relative_db", 0), air.get("problematic", False),
            len(resonances), len(recommandations_eq),
        ])

    # ── Affichage console ──
    _print_analysis(analyse, json_name, csv_name)

    # ── Graphique ──
    if plot_spectre:
        _plot_spectrum(S, frequencies, resonances, zones_critiques, fichier)

    del S  # liberer la memoire de la STFT
    return analyse


def _print_analysis(analyse: dict, json_name: str, csv_name: str) -> None:
    """Affichage synthetique dans la console."""
    # Panel principal
    console.print(Panel.fit(
        f"[bold cyan]Analyse piste : {analyse['fichier']}[/bold cyan]\n"
        f"Duree : [bold]{int(analyse['duree_sec'] // 60)} min {int(analyse['duree_sec'] % 60)} sec[/bold]\n"
        f"TEMPO : [bold]{analyse['tempo']:.1f} BPM[/bold]\n"
        f"RMS : [bold]{analyse['rms']:.4f}[/bold] | Peak : [bold]{analyse['peak']:.4f}[/bold]\n"
        f"LUFS : [bold]{analyse['lufs']:.2f}[/bold] | SNR : [bold]{analyse['snr_db']:.2f} dB[/bold]\n"
        f"Tessiture : [bold]{analyse['tessiture_utile']['min_freq']:.1f}[/bold] - "
        f"[bold]{analyse['tessiture_utile']['max_freq']:.1f} Hz[/bold]\n"
        f"Bande 90% energie : [bold]{analyse['bande_90pct_energie']['freq_low_hz']:.0f}[/bold] - "
        f"[bold]{analyse['bande_90pct_energie']['freq_high_hz']:.0f} Hz[/bold]\n"
        f"Centroid : [bold]{analyse['spectral_centroid_hz']:.1f} Hz[/bold] | "
        f"Rolloff : [bold]{analyse['spectral_rolloff_hz']:.1f} Hz[/bold]\n"
        f"Tonalite : [bold]{analyse['key']}[/bold] | Pitch : [bold]{analyse['pitch_dominant']:.1f} Hz[/bold]\n"
        f"Crest factor : [bold]{analyse['crest_factor']:.2f} dB[/bold]"
    ))

    # Zones critiques
    zones = analyse.get("zones_critiques", {})
    if zones:
        table = Table(title="Zones critiques")
        table.add_column("Zone", style="cyan")
        table.add_column("Relatif (dB)", justify="right")
        table.add_column("Probleme ?", justify="center")
        for zname, zinfo in zones.items():
            status = "[red]OUI[/red]" if zinfo["problematic"] else "[green]NON[/green]"
            table.add_row(zname.capitalize(), f"{zinfo['relative_db']:.1f}", status)
        console.print(table)

    # Resonances
    resonances = analyse.get("resonances_detectees", [])
    if resonances:
        table = Table(title=f"Resonances detectees ({len(resonances)})")
        table.add_column("Freq (Hz)", justify="right", style="yellow")
        table.add_column("Prominence (dB)", justify="right")
        table.add_column("Q estime", justify="right")
        for res in resonances[:10]:
            table.add_row(
                f"{res['frequence_hz']:.1f}",
                f"{res['prominence_db']:.1f}",
                f"{res['q_estime']:.1f}",
            )
        console.print(table)

    # Recommandations EQ
    recs = analyse.get("recommandations_eq", [])
    if recs:
        table = Table(title="Recommandations EQ")
        table.add_column("Freq (Hz)", justify="right", style="magenta")
        table.add_column("Type", justify="center")
        table.add_column("Gain (dB)", justify="right")
        table.add_column("Q", justify="right")
        table.add_column("Raison")
        for rec in recs:
            color = "red" if rec["type"] == "cut" else "green"
            table.add_row(
                f"{rec['frequence_hz']:.0f}",
                f"[{color}]{rec['type'].upper()}[/{color}]",
                f"{rec['gain_db']:.1f}",
                f"{rec['q_suggere']:.1f}",
                rec["raison"],
            )
        console.print(table)
    else:
        console.print("[green]Aucun probleme EQ majeur detecte.[/green]")

    console.print(f"\n[green]JSON : {json_name}[/green]")
    console.print(f"[green]CSV  : analyse_piste.csv[/green]")


def _plot_spectrum(S: NDArray[Any], frequencies: NDArray[Any],
                   resonances: list[dict], zones: dict, fichier: str) -> None:
    """Trace le spectre avec les resonances et zones critiques."""
    energy = np.sum(S ** 2, axis=1)
    energy_db = 10 * np.log10(energy + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogx(frequencies[1:], energy_db[1:], color='steelblue', linewidth=0.8, label='Spectre')

    # Colorer les zones critiques
    zone_colors = {
        "mud":       ("#8B4513", 0.15),
        "boxiness":  ("#FF8C00", 0.15),
        "harshness": ("#DC143C", 0.15),
        "air":       ("#00BFFF", 0.10),
    }
    for zname, (fmin, fmax) in CRITICAL_ZONES.items():
        color, alpha = zone_colors[zname]
        prob = zones.get(zname, {}).get("problematic", False)
        if prob:
            alpha = 0.3
        ax.axvspan(fmin, fmax, alpha=alpha, color=color, label=f"{zname.capitalize()}")

    # Marquer les resonances
    for res in resonances[:10]:
        ax.axvline(res["frequence_hz"], color='red', alpha=0.5, linewidth=0.8, linestyle='--')

    ax.set_xlabel("Frequence (Hz)")
    ax.set_ylabel("Energie (dB)")
    ax.set_title(f"Analyse spectrale - {os.path.basename(fichier)}")
    ax.set_xlim(20, 20000)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()


def batch_analyse_piste(folder: str, plot_spectre: bool = False, output_dir: str = "",
                        duration: float | None = None) -> None:
    """Analyse batch de toutes les pistes d'un dossier.
    Si WAV et MP3 existent pour le meme stem, seul le WAV est analyse.
    """
    audio_exts = (".wav", ".mp3", ".flac", ".aiff", ".ogg")
    all_files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(audio_exts)]
    # Deduplication : preferer WAV sur MP3 quand les deux existent
    stems: dict[str, str] = {}
    priority = {".wav": 0, ".flac": 1, ".aiff": 2, ".mp3": 3, ".ogg": 4}
    for fname in all_files:
        stem = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1].lower()
        if stem not in stems or priority[ext] < priority[os.path.splitext(stems[stem])[1].lower()]:
            stems[stem] = fname
    files = [os.path.join(folder, fname) for fname in sorted(stems.values())]
    if not files:
        console.print(f"[yellow]Aucun fichier audio dans {folder}[/yellow]")
        return
    for f in files:
        console.print(f"\n[cyan]--- {os.path.basename(f)} ---[/cyan]")
        try:
            analyse_piste(f, plot_spectre=plot_spectre, output_dir=output_dir, duration=duration)
        except Exception as e:
            console.print(f"[red]Erreur sur {f} : {e}[/red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse de piste individuelle (kick, snare, voix, basse, synthe)",
        epilog="Exemples:\n"
               "  python analyse_piste.py kick.wav\n"
               "  python analyse_piste.py -f vocals.wav --plot\n"
               "  python analyse_piste.py -d ./stems/\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fichier", nargs="?", help="Fichier audio a analyser")
    parser.add_argument("-f", "--file", help="Fichier audio a analyser (avec option)")
    parser.add_argument("-d", "--dir", help="Dossier a analyser en batch")
    parser.add_argument("--plot", action="store_true", help="Afficher le graphique spectral")
    parser.add_argument("-o", "--output", default="", help="Dossier de sortie pour les JSON/CSV (defaut: dossier courant)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Analyser seulement les N premieres secondes (ex: 120). Defaut: fichier entier")
    args = parser.parse_args()

    fichier_cible = args.fichier or args.file

    if args.duration:
        console.print(f"[dim]Duree limitee a {args.duration:.0f} secondes par piste[/dim]")

    if fichier_cible:
        if os.path.exists(fichier_cible):
            console.print(f"[cyan]Analyse piste : {fichier_cible}[/cyan]\n")
            analyse_piste(fichier_cible, plot_spectre=args.plot, output_dir=args.output,
                          duration=args.duration)
        else:
            console.print(f"[red]Fichier introuvable : {fichier_cible}[/red]")
    elif args.dir:
        if os.path.exists(args.dir):
            console.print(f"[cyan]Analyse batch : {args.dir}[/cyan]\n")
            batch_analyse_piste(args.dir, plot_spectre=args.plot, output_dir=args.output,
                                duration=args.duration)
        else:
            console.print(f"[red]Dossier introuvable : {args.dir}[/red]")
    else:
        console.print("[yellow]Utilisation :[/yellow]")
        console.print("  [green]python analyse_piste.py kick.wav[/green]")
        console.print("  [green]python analyse_piste.py -f vocals.wav --plot[/green]")
        console.print("  [green]python analyse_piste.py -d ./stems/[/green]")
        console.print("\n[dim]--help pour plus d'informations[/dim]")
