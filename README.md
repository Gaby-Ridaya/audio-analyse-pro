# Audio Analyse Pro

Outils Python d'**analyse audio professionnelle** orientée mixage et mastering studio.

Deux scripts indépendants couvrent deux besoins distincts :
- [`audio_analyse.py`](#audio_analysepy--analyse-mixmastering) — analyse complète d'un mix ou d'un master
- [`analyse_piste.py`](#analyse_pistepy--analyse-piste-individuelle-eq--résonances) — analyse EQ d'une piste séparée (kick, snare, voix, basse…)

---

## `audio_analyse.py` — Analyse mix/mastering

Analyse approfondie d'un fichier audio avec :

| Fonctionnalité | Détail |
|---|---|
| **LUFS intégrée** | ITU-R BS.1770-4, double gating absolu + relatif |
| **True Peak** | Oversampling ×4, conforme BS.1770 |
| **LRA** | Loudness Range en LU (fenêtres 3 s / hop 1 s) |
| **Analyse Mid/Side** | Ratio M/S dB, corrélation L/R, stereo width |
| **Features spectrales** | Centroid, rolloff 95 %, flatness, contrast 7 bandes |
| **Énergie par bande** | Grave, bas-médium, médium, haut-médium, aigu |
| **Comparaison référence** | Delta LUFS, crest, centroid, True Peak, M/S + conseils |
| **Interprétation auto** | Conseils mixage (compression, loudness, balance, stéréo) |
| **Optimisation mémoire** | Traitement streaming par blocs 30 s (~15 MB/bloc) |
| **Export** | JSON (par fichier) + CSV cumulatif |

### Utilisation

```bash
# Fichier unique
python audio_analyse.py mon_mix.wav

# Avec graphique spectral
python audio_analyse.py -f mon_mix.wav --plot

# Analyse batch d'un dossier
python audio_analyse.py -d ./audio/

# Comparaison avec une référence
python audio_analyse.py --compare mon_mix.wav --ref reference.wav
```

### Exemple de sortie console

```
╭──────────────────────────────────────────────────────────────────╮
│ Analyse pro : mon_mix.wav                                        │
│ Durée : 3 min 45 sec                                             │
│ TEMPO : 112.3 BPM                                                │
│ RMS : 0.1011 | Peak : 0.4621                                     │
│ LUFS : -14.20 | True Peak : -6.71 dBTP                           │
│ LRA : 8.50 LU | SNR : 37.47 dB                                   │
│ Centroid : 628.6 Hz | Rolloff 95% : 2863.9 Hz                    │
│ M/S Ratio : -4.74 dB | Stereo Width : 0.5027 | Corr : 0.4973    │
│ Crest factor : 13.20 dB | Dynamique RMS : 0.0177                 │
╰──────────────────────────────────────────────────────────────────╯

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Interpretation & Conseils Mixage                           │
│ Parametre     │ Valeur  │ Diagnostic               │ Conseil                    │
│ crest_factor  │ 13.20   │ Bonne dynamique          │ Idéal jazz/acoustique      │
│ lufs          │ -20.99  │ Mix tranquille            │ Acceptable, contenu dyn.   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## `analyse_piste.py` — Analyse piste individuelle (EQ + résonances)

Analyse dédiée aux pistes séparées avec :

| Fonctionnalité | Détail |
|---|---|
| **Résonances étroites** | Détection via STFT + `scipy.find_peaks`, top 15, Q estimé |
| **Bande 90 % énergie** | Fréquences basse/haute contenant 90 % de l'énergie |
| **Zones critiques** | Mud 200-400 Hz, Boxiness 400-800 Hz, Harshness 2-5 kHz, Air 8-15 kHz |
| **Recommandations EQ** | Fréquence, type cut/boost, gain dB, Q suggéré, raison |
| **Détection silence DAW** | Saute automatiquement le silence de timeline en début de fichier |
| **Analyse batch** | Dossier complet avec déduplication WAV > MP3 |
| **Graphique** | Spectre avec zones colorées et marqueurs de résonances |
| **Export** | JSON (par piste) + CSV cumulatif |

### Utilisation

```bash
# Piste unique
python analyse_piste.py kick.wav

# Avec graphique spectral
python analyse_piste.py -f vocals.wav --plot

# Analyse batch d'un dossier de stems
python analyse_piste.py -d ./stems/

# Limiter l'analyse aux N premières secondes
python analyse_piste.py kick.wav --duration 60

# Spécifier un dossier de sortie
python analyse_piste.py -f bass.wav -o ./resultats/
```

### Exemple de sortie console

```
┌──────────────────────────────────────┐
│          Zones critiques             │
│ Zone       │ Relatif (dB) │ Problème │
│ Mud        │ +4.2         │ OUI      │
│ Boxiness   │ +1.8         │ NON      │
│ Harshness  │ -0.5         │ NON      │
│ Air        │ -5.1         │ OUI      │
└──────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       Recommandations EQ                            │
│ Freq (Hz) │ Type  │ Gain (dB) │ Q   │ Raison                       │
│ 300       │ CUT   │ -1.7      │ 1.5 │ Accumulation Mud (200-400 Hz) │
│ 12000     │ BOOST │ +1.5      │ 0.7 │ Manque d'air (8-15 kHz)      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
pip install -r requirements.txt
```

| Paquet | Rôle |
|--------|------|
| `librosa` | Analyse audio (STFT, chroma, onset, beat) |
| `soundfile` | Lecture streaming par blocs |
| `pyloudnorm` | LUFS / K-weighting |
| `scipy` | Filtres IIR, détection de pics |
| `numpy` | Calcul numérique |
| `rich` | Affichage console coloré |
| `matplotlib` | Graphiques spectraux |

**Python 3.10+** requis (annotations `int | float` et `list[dict]`).

---

## Formats audio supportés

WAV, MP3, FLAC, AIFF, OGG

---

## Fichiers de sortie

| Outil | JSON | CSV |
|-------|------|-----|
| `audio_analyse.py` | `<nom_fichier>.json` | `analyse_audio.csv` (cumulatif) |
| `analyse_piste.py` | `<nom_fichier>_piste.json` | `analyse_piste.csv` (cumulatif) |

`analyse_audio.csv` inclus dans ce dépôt comme exemple de données de sortie.

---

## Licence

MIT
