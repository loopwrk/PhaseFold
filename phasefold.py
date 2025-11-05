# A generative sound and visual synthesis tool that creates evolving harmonic 
# textures based on recursive processes.
# Dependencies: numpy, matplotlib, sounddevice, tkinter

import sys
import os

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
import numpy as np
import threading
from pathlib import Path
import wave
import json

# --- Dependency check (first-run universal setup) ---
# Check for required dependencies before importing them
_required_modules = [
    "numpy",
    "matplotlib",
]
_missing = []
for _mod in _required_modules:
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_mod)
if _missing:
    print("❌ Missing dependencies detected:")
    for name in _missing:
        print(f"  - {name}")
    print("Please install them manually with:")
    print("    python3 -m pip install -r requirements.txt")
    sys.exit(1)

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 0.15
matplotlib.rcParams["agg.path.chunksize"] = 2000
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Application constants ---
AUDIO_SR = 44100
CONTROL_HZ = 60.0
LISSAJOUS_POINTS = 1100 # ~50 ms window (≈1100 pts at 44.1 kHz, about half of 2205)
WAVEFORM_PLOT_POINTS = 1000
VIZ_WINDOW_SECONDS = 0.05  # 50 ms visualization window

try:
    import sounddevice as sd  # primary, low-latency
except Exception:
    sd = None
    
# ----------------------- Utility functions -----------------------

def write_wav_stereo(path, L, R, sr):
    path = Path(path)
    xL = np.clip(L, -1.0, 1.0)
    xR = np.clip(R, -1.0, 1.0)
    pcm = (np.stack([xL, xR], axis=1) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path

# ----------------------- DSP helpers -----------------------

def freq_to_note_name(freq):
    """Convert frequency in Hz to note name (e.g., 'A4', 'C#3')."""
    if freq <= 0:
        return "---"

    # A4 = 440 Hz reference
    A4 = 440.0
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Calculate semitones from A4
    semitones_from_A4 = 12 * np.log2(freq / A4)
    # Round to nearest semitone
    semitone_index = int(round(semitones_from_A4))

    # A4 is note index 9 (A) in octave 4
    # Calculate which note and octave
    # A4 = 0 semitones, A#4 = 1, B4 = 2, C5 = 3, etc.
    # A3 = -12, A2 = -24, etc.
    note_index = (9 + semitone_index) % 12  # 9 = A in the note_names list
    octave = 4 + (9 + semitone_index) // 12

    return f"{note_names[note_index]}{octave}"


def lowpass(sig, cutoff_hz, sr, state=0.0):
    """
    One-pole low-pass filter: y[n] = y[n-1] + α (x[n] - y[n-1])
    where α = 1 - exp(-2π fc / sr)

    This exponential form is numerically stable and avoids division by zero.

    Args:
        sig: Input signal array
        cutoff_hz: Cutoff frequency in Hz
        sr: Sample rate in Hz
        state: Previous output sample for continuity across chunks (default: 0.0)

    Returns:
        Filtered signal (or bypass if cutoff is out of range)

    Edge cases:
        - cutoff_hz <= 0: bypass (return copy of input)
        - cutoff_hz >= sr/2: bypass (already at Nyquist)
        - non-finite cutoff_hz: bypass
    """
    sig = np.asarray(sig)
    if sig.size == 0:
        return sig

    # Bypass filter if cutoff is invalid or out of usable range
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0 or cutoff_hz >= 0.5 * sr:
        return sig.copy()

    # Exponential coefficient form: stable, no division by tiny numbers
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sr)

    y = np.empty_like(sig)
    acc = float(state)
    for i in range(sig.shape[0]):
        acc += alpha * (sig[i] - acc)
        y[i] = acc
    return y

# Simplex‑safe stabilizer for state vector [m,progress_01]
_def_eps = 1e-12

# Stabilizes a 2-element state vector by clamping it, mapping through a sigmoid (±8 covers >99.999% of its range),
# and normalizing it so components sum to 1. Keeps recursive dynamics numerically stable
# and ensures values stay within a valid probability-like simplex.
def stabilize_state(v):
    v = np.clip(v, -8.0, 8.0)
    v01 = 1.0 / (1.0 + np.exp(-v))
    s = float(v01[0] + v01[1]) + _def_eps
    return v01 / s

# A retrocausal constraint that projects the state vector toward the "unified" state.
# Used to model convergence during recursive evolution.
def proj_P(v):
    v = stabilize_state(v)
    s = float(v[0] + v[1])
    return np.array([s, 0.0])

def mix_R(theta):
    # Bistochastic mixing matrix to stay on simplex
    c = 0.5 * (1.0 + np.cos(theta))
    s = 0.5 * (1.0 + np.sin(theta))
    return np.array([[c, s], [1.0 - c, 1.0 - s]])

# Applies an asymmetry tilt to the state vector using epsilon.
def tilt_A(eps):
    e = float(np.clip(eps, 0.0, 0.25))
    return np.array([[1.0, e], [0.0, 1.0 - e]])

# Core recursive transformation (Φ) for evolving the system state.
# Combines projection, tilt, and rotation steps:
#   1. Blend current state with its projection (λ controls convergence)
#   2. Apply asymmetry tilt (ε)
#   3. Apply rotational mix (θ_step)
# Returns the updated, renormalized 2-element state vector.
def apply_Phi(v, lam, theta_step, eps):
    lam = float(np.clip(lam, 0.0, 1.0))
    theta = float(np.clip(theta_step, -0.05, 0.05))
    v = stabilize_state(v)
    v1 = (1.0 - lam) * v + lam * proj_P(v)
    v2 = tilt_A(eps) @ v1
    v3 = mix_R(theta) @ v2
    v3 = np.clip(v3, 1e-12, 1.0)
    return v3 / float(v3[0] + v3[1])

# ----------------------- Core generator -----------------------

def generate_app(
    dur=20.0,
    sr=AUDIO_SR,
    base_f0=110.0,
    voices=6,
    layers=4,
    stereo_width=0.35,
    seed=2025,
    fm_index0=1.6,
    am_index0=0.35,
    collapse_curve=2.2,
    binaural_delta_hz0=7.0,
    binaural_amount=0.22,
    overtone_power=1.3,
    harmonic_even=0.20,
    harmonic_odd=0.16,
    comb_amount=0.10,
    voice_delay=0.0,
    breath_rate=0.045,
):
    # Make time axis
    SAMPLE_TOTAL = int(sr * dur)
    sample_times_s = np.arange(SAMPLE_TOTAL) / sr

    progress_01 = np.linspace(0, 1, SAMPLE_TOTAL)
    
    # Convergence envelope over time: start fully active (1.0) and decay toward 0
    # Shape is set by collapse_curve: higher values = hold longer, then drop faster
    # Used to fade motion, binaural spread, and harmonic energy back into the base tone
    convergence_gain_01 = 1.0 - progress_01**collapse_curve

    # Voice emergence envelope: delay voices, fade them in over ~1-2s
    voice_delay_samps = int(voice_delay * sr)
    voice_fade_samps = int(1.5 * sr)  # 1.5s fade-in
    voice_emerge_env = np.ones(SAMPLE_TOTAL)
    if voice_delay_samps > 0:
        # Before delay: voices silent
        voice_emerge_env[:voice_delay_samps] = 0.0
        # After delay: fade in
        fade_end = min(SAMPLE_TOTAL, voice_delay_samps + voice_fade_samps)
        fade_length = fade_end - voice_delay_samps
        if fade_length > 0:
            voice_emerge_env[voice_delay_samps:fade_end] = np.linspace(
                0, 1, fade_length
            )

    # Create a small, natural-sounding detune for each voice:
    # 1. Initialize a reproducible random number generator with the given seed.
    # 2. Generate random detune offsets in cents (normally distributed around 0, ±12 cents typical range).
    # 3. Convert these cent values into frequency ratios (detune factors) using the equal-temperament formula:
    #       detune_factor = 2 ** (cents / 1200)
    #    but scale them per-layer by the current convergence gain (so deeper recursion = more subtle detune).
    # 4. Randomize the starting phase of each oscillator between 0 and 2π, giving each voice a unique phase start.
    rng = np.random.default_rng(seed)
    cents = rng.normal(0.0, 12.0, size=voices)
    detune_factor = 2 ** ((cents[:, None] / 1200.0) * convergence_gain_01[None, :])
    phase0 = rng.uniform(0, 2 * np.pi, size=voices)

    # Breath oscillation - make it more prominent when breath_rate > 0
    if breath_rate > 0:
        breath = 0.5 * (1 + np.sin(2 * np.pi * breath_rate * sample_times_s))
        # Scale breath intensity: stronger at beginning, fades with convergence_gain_01
        breath = breath * (0.3 + 0.7 * convergence_gain_01)
    else:
        breath = np.ones(SAMPLE_TOTAL) * 0.5  # neutral when disabled

   # Recursion control rate
    control_hz = CONTROL_HZ
    Nc = max(2, int(dur * control_hz))
    recursion_ctrl_progress = np.linspace(0.0, 1.0, Nc)
    convergence_gain_ctrl_01 = 1.0 - recursion_ctrl_progress**collapse_curve
    
    # tilt_amplitude: magnitude of the bias (ε) applied each recursion step.
    # Controls how strongly each iteration deviates from symmetry.
    tilt_amplitude = 0.08 * convergence_gain_ctrl_01

    v_state_c = np.zeros((2, Nc))
    v_state_c[:, 0] = stabilize_state(np.array([0.0, 1.0]))
    for i in range(1, Nc):
        theta_step_i = (2 * np.pi * (0.05 + 0.10 * convergence_gain_ctrl_01[i])) / control_hz
        v_state_c[:, i] = apply_Phi(
            v_state_c[:, i - 1], convergence_gain_ctrl_01[i], theta_step_i, tilt_amplitude[i]
        )

    # Control‑rate → audio‑rate
    # Filter slow processes at control-rate for better performance.
    marked_state_c = v_state_c[0]
    mclip_c = np.clip(marked_state_c, -8.0, 8.0)
    activity_env_c = 0.5 * (1.0 + np.tanh(0.5 * 3.0 * mclip_c))
    # Filter slow envelope at control‑rate (cutoff 0.5 Hz, sr = control_hz)
    activity_env_c = lowpass(activity_env_c, cutoff_hz=0.5, sr=control_hz)
    # Upsample back to audio‑rate for application in the signal domain
    activity_env = np.interp(progress_01, recursion_ctrl_progress, activity_env_c)

    # Audio‑rate marked_state (for tiny pitch drift) derived from control‑rate state
    marked_state = np.interp(progress_01, recursion_ctrl_progress, marked_state_c)
    freq_dev = 0.01 * (marked_state - np.mean(marked_state))
    amp_env = 0.6 + 0.4 * activity_env

    base_f = base_f0 * (1.0 + freq_dev)
    
    # --- Layering model ---
    # Synthesize one submix per layer with slower envelopes and depth-scaled FM/AM,
    # then mix layers with collapse-aware weight.

    # Per-layer control envelopes at control-rate (deeper layers evolve more slowly)
    # higher layers amplify this via the triangular factor, so we start conservatively.

    # Per-layer control envelopes at control-rate (deeper layers evolve more slowly)
    # Slowdown factor controls how much each deeper layer's envelope is slowed compared to the surface layer
    # 0.75 makes each successive layer roughly 25–40% slower than the last — enough to feel distinctly
    # deeper, but still breathing in sync
    LAYER_CTRL_SLOWDOWN = 0.75
    base_fc = 1.0  # Hz at control-rate
    layer_ctrl_c = []
    for ell in range(layers):
        fc_ell = base_fc / (1.0 + LAYER_CTRL_SLOWDOWN * ell)
        env_c = lowpass(activity_env_c, cutoff_hz=fc_ell, sr=control_hz)
        layer_ctrl_c.append(env_c)
    # Upsample to audio-rate
    layer_ctrl = [np.interp(progress_01, recursion_ctrl_progress, e) for e in layer_ctrl_c]

    # Slow drift coefficient shared across layers (keeps the deep fold memory)
    drift_coeff = 0.02 * (layers * (layers + 1) / 2.0)
    drift_phase = 2 * np.pi * (drift_coeff * convergence_gain_01) * sample_times_s

    # Depth scalers: deeper layers subtler (more stable)
    alpha_fm = 0.82
    beta_am  = 0.88

    # Tiny depth detune so layers don’t phase-cancel perfectly
    depth_cents = 3.0
    layer_detune = [2 ** (((ell - 0.5 * (layers - 1)) * depth_cents) / 1200.0) for ell in range(layers)]

    layer_sums = []
    for ell in range(layers):
        ctrl_l = layer_ctrl[ell]
        fm_l = np.minimum(fm_index0 * (alpha_fm ** ell) * convergence_gain_01, 0.6)
        am_l = np.minimum(am_index0 * (beta_am  ** ell) * convergence_gain_01, 0.4)

        # Apply small depth detune to the per-voice base frequency
        phase_layer = (
            2 * np.pi * np.cumsum(detune_factor * (base_f[None, :] * layer_detune[ell]), axis=1) / sr
        ) + phase0[:, None]

        # FM modulation from this layer’s control envelope
        phase_mod_l = 2 * np.pi * np.cumsum(ctrl_l * fm_l) / sr

        # Compose phases (include shared slow drift)
        phase_l = phase_layer + drift_phase[None, :] + phase_mod_l[None, :]

        # AM from this layer
        amp_l = (1 - am_l) + am_l * ctrl_l
        y_l = np.sin(phase_l) * amp_l

        # Submix across voices for this layer
        midsum_l = np.sum(y_l, axis=0) / voices
        layer_sums.append(midsum_l)

    # Collapse-aware weighting: deeper layers dominate later
    w_raw = np.stack([activity_env ** (ell + 1) for ell in range(layers)], axis=0)
    w_sum = np.maximum(1e-9, np.sum(w_raw, axis=0, keepdims=True))
    w = w_raw / w_sum

    stacked = np.stack(layer_sums, axis=0)
    mixed_signal = np.sum(w * stacked, axis=0)

    # Bbase tone anchors unity, voices fade in
    base_phase = 2 * np.pi * base_f0 * np.arange(1, SAMPLE_TOTAL + 1) / sr
    base_tone_core = np.sin(base_phase) * amp_env * (0.5 + 0.5 * breath)
    base_gain = 1.0 + 0.15 * (1.0 - voice_emerge_env) - 0.10 * (
        voice_emerge_env * convergence_gain_01
    )
    base_gain = np.clip(base_gain, 0.75, 1.25)
    base_tone = base_tone_core * base_gain

    # Combine base with layer submix; keep the same gentle nonlinearity
    mixed_signal = base_tone + (mixed_signal - base_tone) * voice_emerge_env
    mix = np.tanh(0.9 * mixed_signal)

    # Overtones (T2/T3) and comb (tied to early activity via convergence_gain_01)
    x = np.clip(mix, -1.0, 1.0)
    even_term = 2.0 * x * x - 1.0
    odd_term = 4.0 * x * x * x - 3.0 * x
    mix = np.tanh(
        mix
        + (convergence_gain_01**overtone_power)
        * (harmonic_even * even_term + harmonic_odd * odd_term)
    )

    if comb_amount > 0:
        d1 = int(sr / base_f0)
        d2 = int(sr / (base_f0 * 1.5))
        d3 = int(sr / (base_f0 * 2.0))
        acc = np.zeros_like(mix)
        if d1 > 0:
            acc[:-d1] += mix[d1:]
        if d2 > 0:
            acc[:-d2] += mix[d2:]
        if d3 > 0:
            acc[:-d3] += mix[d3:]

        # Apply short fade-in to acc to prevent clicks from abrupt onset
        # 20ms is short enough to be inaudible but long enough to smooth discontinuities
        fade_len = min(int(0.02 * sr), len(acc))  # 20ms fade-in
        if fade_len > 1:
            fade_in = np.linspace(0, 1, fade_len)
            acc[:fade_len] *= fade_in

        aenv = convergence_gain_01**overtone_power
        mix = np.tanh(
            (1.0 - comb_amount * 3.0 * aenv) * mix + (comb_amount * aenv) * acc
        )

    # Simple, fast stereo + binaural collapse
    binaural_env = convergence_gain_01**1.4
    delta_t = binaural_delta_hz0 * binaural_env
    phase_b_L = 2 * np.pi * np.cumsum(base_f0 - 0.5 * delta_t) / sr
    phase_b_R = 2 * np.pi * np.cumsum(base_f0 + 0.5 * delta_t) / sr
    b_L = np.sin(phase_b_L)
    b_R = np.sin(phase_b_R)

    # Increased breath effect in stereo field: was 0.75/0.25, now 0.6/0.4 for more audible pulsing
    L_base = mix * (0.6 + 0.4 * breath) + binaural_amount * binaural_env * b_L
    R_base = mix * (0.6 + 0.4 * (1 - breath)) + binaural_amount * binaural_env * b_R

    st_env = 0.5 * stereo_width * convergence_gain_01
    delay_samps = 48
    delayed = np.empty_like(mix)
    delayed[:delay_samps] = 0.0
    delayed[delay_samps:] = mix[:-delay_samps]
    
    L = L_base + st_env * delayed
    R = R_base - st_env * delayed

    # Apply fade in at start
    fade_in_s = 0.5
    fi = int(fade_in_s * sr)
    fade_up = np.linspace(0, 1, fi)
    L[:fi] *= fade_up
    R[:fi] *= fade_up


    # Let the recursion decide when to end. We measure how fast the control‑rate
    # envelope is changing; when it’s essentially still for a while, we "detect 
    # collapse" and gently stop.

    # Measure control‑rate activity (|d/dt| of activity_env_c), smoothed at control_hz
    d_ctrl = np.abs(np.diff(activity_env_c, prepend=activity_env_c[0])) * control_hz
    d_ctrl_smooth = lowpass(d_ctrl, cutoff_hz=0.5, sr=control_hz)

    # Threshold: when smoothed change stays below eps for quiet_secs, we stop soon after
    eps = 1e-3          # small envelope change per second considered "still"
    quiet_secs = 2.0    # require this long of stillness at control-rate
    quiet_steps = max(1, int(round(quiet_secs * control_hz)))

    # Find the last index where activity was above threshold; stop after a quiet window
    active_idx = np.where(d_ctrl_smooth > eps)[0]

    if active_idx.size > 0:
        last_active = int(active_idx[-1])
        stop_ctrl_idx = min(Nc - 1, last_active + quiet_steps)
    else:
        # No activity at all → allow full piece, stop at the end
        stop_ctrl_idx = Nc - 1

    # Map control index → audio sample index
    stop_t = recursion_ctrl_progress[stop_ctrl_idx]
    stop_idx = int(round(stop_t * SAMPLE_TOTAL))
    stop_idx = max(1, min(stop_idx, len(mix)))

    # Build final stereo from mix buffers up to stop point
    L = L[:stop_idx].copy()
    R = R[:stop_idx].copy()

    # Apply a short equal‑power fade to silence so we never click
    fade_out_duration = 1.0  # seconds; short, natural
    fade_samps = max(1, min(int(round(fade_out_duration * sr)), len(L)))
    if fade_samps > 0:
        f = np.cos(0.5 * np.pi * np.linspace(0.0, 1.0, fade_samps))
        L[-fade_samps:] *= f
        R[-fade_samps:] *= f

    # Optional: trim trailing near‑silence with a 50 ms safety pad for visual sync
    mono_abs = 0.5 * (np.abs(L) + np.abs(R))
    thr = 5e-4  # about −66 dBFS
    nz = np.where(mono_abs > thr)[0]
    if nz.size > 0:
        cut = min(len(L), int(nz[-1]) + 1 + int(VIZ_WINDOW_SECONDS * sr))
        L = L[:cut]
        R = R[:cut]

    # Fallback safety: if something went odd and we ended absurdly early (< 30% of target),
    # keep the original length but still apply a fade at the nominal end to avoid surprise.
    if len(L) < int(0.30 * SAMPLE_TOTAL):
        L = L_base.copy()
        R = R_base.copy()
        # Reconstruct stereo with delay contribution just like above window
        delayed = np.empty_like(mix)
        delayed[:delay_samps] = 0.0
        delayed[delay_samps:] = mix[:-delay_samps]
        L += st_env * delayed
        R -= st_env * delayed
        # Apply a graceful 1 s fade at the very end
        fade_samps = max(1, min(int(round(1.0 * sr)), len(L)))
        f = np.cos(0.5 * np.pi * np.linspace(0.0, 1.0, fade_samps))
        L[-fade_samps:] *= f
        R[-fade_samps:] *= f

    # headroom normalize
    peak = max(np.max(np.abs(L)), np.max(np.abs(R))) + 1e-12
    gain = (10 ** (-1.5 / 20)) / peak
    return L * gain, R * gain, sr, recursion_ctrl_progress, d_ctrl_smooth


# ----------------------- Lissajous widget -----------------------


class LissajousPane(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.show_axes = True  # Toggle for axes/labels
        self.auto_scale = False  # Toggle for auto-scaling to fill view
        self.padding = 0.1  # Padding when auto-scale is enabled (10%)

        # Gradient color cycling through preset palette
        self.color_phase = 0.0
        self._palette_hex = ["#5700EE", "#FF0683", "#0042ED", "#0000FF", "#ff0400", "#0900ED", "#894FED", "#6E2BFF", "#9500FF", "#1A4CFF", "#FF33B1", "#232EFF", "#A070FF"]
        self._palette_rgb = [self._hex_to_rgb(c) for c in self._palette_hex]
        self._palette_len = len(self._palette_rgb)
        self.stroke_width = 1

        # MPL point budget (max number of XY points per frame).
        # Keep visuals crisp while reducing per-frame segment count.
        self._mpl_points = LISSAJOUS_POINTS  # ~half of the typical 2205-window

        # Cache for latest frame (so audio path can just drop data here)
        self._latest_xy = None
        # Throttle draws to protect audio
        self._last_draw_t = 0.0
        self._mpl_min_interval = 1.0 / 30.0  # 30 FPS max

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_mpl.get_tk_widget().pack(fill="both", expand=True)
        (self.line,) = self.ax.plot([], [], linewidth=self.stroke_width)
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("Left → X")
        self.ax.set_ylabel("Right → Y")
        self.ax.set_title("Lissajous (XY)")
        try:
            self.ax.set_facecolor("#f6f5f2")
            self.fig.patch.set_facecolor("#f6f5f2")
        except Exception:
            pass

    def set_display_options(self, show_axes=True, auto_scale=False):
        self.show_axes = bool(show_axes)
        self.auto_scale = bool(auto_scale)

        ax = self.ax
        # Labels/title
        if self.show_axes:
            ax.set_xlabel("Left → X")
            ax.set_ylabel("Right → Y")
            ax.set_title("Lissajous (XY)")
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

        # Spines and ticks in one go
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(self.show_axes)
        ax.tick_params(left=self.show_axes, bottom=self.show_axes,
                       labelleft=self.show_axes, labelbottom=self.show_axes)

        self.canvas_mpl.draw_idle()

    def _hex_to_rgb(self, hex_color):
        """Parse #RRGGBB into integer RGB tuple."""
        value = hex_color.strip().lstrip("#")
        if len(value) != 6:
            raise ValueError(f"Unsupported hex color '{hex_color}'")
        return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))

    def _gradient_color(self, phase):
        """Return a hex color interpolated along the palette for the given phase.
        Robust to floating-point wrap and degenerate palette sizes.
        """
        n = int(getattr(self, "_palette_len", 0))
        if n <= 0:
            return "#ffffff"
        if n == 1:
            c = self._palette_rgb[0]
            return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

        # Map phase into [0, n) but guard against rare t == n due to FP error.
        m = float(n)
        t = float(phase) % m
        if not np.isfinite(t):
            t = 0.0
        if t >= m - 1e-9:
            t = 0.0

        base_idx = int(np.floor(t))  # 0..n-1
        next_idx = (base_idx + 1) % n
        local_t = float(t - base_idx)

        c0 = self._palette_rgb[base_idx]
        c1 = self._palette_rgb[next_idx]
        r = int(round(c0[0] + (c1[0] - c0[0]) * local_t))
        g = int(round(c0[1] + (c1[1] - c0[1]) * local_t))
        b = int(round(c0[2] + (c1[2] - c0[2]) * local_t))
        return f"#{r:02x}{g:02x}{b:02x}"


    def _advance_color_phase(self, step):
        """Advance the animation phase while keeping it bounded."""
        if self._palette_len == 0:
            return
        self.color_phase = (self.color_phase + step) % self._palette_len

    def set_stroke_width(self, width):
        """Update the stroke width for MPL rendering."""
        width = float(max(0.4, min(2.0, width)))
        if abs(self.stroke_width - width) < 1e-6:
            return
        self.stroke_width = width
        if hasattr(self, "line"):
            self.line.set_linewidth(width)
            self.canvas_mpl.draw_idle()
            return

    def show_frame(self, x):
        """
        Lightweight entrypoint: called often (maybe from playback/UI refresh).
        We only STORE the latest frame here to avoid doing heavy MPL work too often.
        Actual drawing happens in _draw_cached_frame(), which is called from the UI thread.
        """
        if x is None or len(x) == 0:
            self._latest_xy = None
            return

        # Store the raw stereo frame
        self._latest_xy = x

    def set_target_fps(self, fps):
        """Set visual refresh rate, hard-capped at 60 FPS, and adapt point budget for stability."""
        try:
            tfps = int(fps)
        except Exception:
            tfps = 30
        tfps = max(10, min(tfps, 60))  # hard cap @ 60 FPS
        self._mpl_min_interval = 1.0 / float(tfps)

    def _draw_cached_frame(self):
        """
        Actually render the latest cached frame to MPL.
        Intended UI-thread rate: ~20–60 FPS (hard cap at 60 via _mpl_min_interval).
        """
        buf = self._latest_xy
        if buf is None:
            # Nothing to draw, but still refresh MPL so it doesn't freeze
            self.line.set_data([], [])
            self.canvas_mpl.draw_idle()
            return

        L = buf[:, 0]
        R = buf[:, 1]

        # Choose scaling
        if self.auto_scale:
            m = max(1e-9, float(np.max(np.abs([L, R]))))
            scale = 1.0 / (m * (1.0 + self.padding))
            X = L * scale
            Y = R * scale
        else:
            m = max(1e-9, float(np.max(np.abs([L, R]))))
            X = L / m
            Y = R / m

        # Downsample to MPL budget
        try:
            max_mpl_pts = int(getattr(self, "_mpl_points", LISSAJOUS_POINTS))
            n_pts = int(X.size)
            if n_pts > max_mpl_pts and max_mpl_pts > 4:
                idx = np.linspace(0, n_pts - 1, max_mpl_pts)
                X = np.interp(idx, np.arange(n_pts), X)
                Y = np.interp(idx, np.arange(n_pts), Y)
        except Exception:
            pass

        # Throttle actual draw
        import time as _time
        now = _time.perf_counter()
        min_interval = getattr(self, "_mpl_min_interval", 1.0 / 30.0)
        last_t = getattr(self, "_last_draw_t", 0.0)
        if (now - last_t) < min_interval:
            # Don't draw yet, just return — next UI tick will try again
            return
        self._last_draw_t = now

        # Draw
        base_col = self._gradient_color(self.color_phase)
        self._advance_color_phase(0.4 / 6.0)
        self.line.set_color(base_col)
        self.line.set_linewidth(self.stroke_width)
        self.line.set_data(X, Y)
        self.canvas_mpl.draw_idle()



# ----------------------- Waveform/Spectrogram widget -----------------------


class WaveformPane(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda e: self._request_redraw())
        self._last_data = None

        # Canvas item reuse
        self._wave_id = None
        self._mid_id = None
        self._last_h = None
        self._redraw_pending = False

    def show_frame(self, x):
        if x is None or len(x) == 0:
            self._last_data = None
        else:
            self._last_data = x
        self._request_redraw()

    def _request_redraw(self):
        if self._redraw_pending:
            return
        self._redraw_pending = True
        try:
            self.after_idle(self._perform_redraw)
        except Exception:
            # Fallback to immediate redraw if scheduling fails
            self._perform_redraw()

    def _perform_redraw(self):
        self._redraw_pending = False
        self._redraw()

    def _redraw(self):
        c = self.canvas
        try:
            if not self.winfo_exists() or not c.winfo_exists():
                return
        except Exception:
            return
        if self._last_data is None:
            return

        w = max(10, c.winfo_width())
        h = max(10, c.winfo_height())

        # Guard against invalid dimensions
        if w <= 1 or h <= 1:
            return

        mid_y = h // 2

        # --- Waveform ---
        x = self._last_data
        if len(x.shape) == 2:
            mono = np.mean(x, axis=1)  # mix to mono for display
        else:
            mono = x

        n = len(mono)
        if n == 0:
            return

        fixed_points = WAVEFORM_PLOT_POINTS
        if n < fixed_points:
            # Pad with zeros if input is too short
            mono = np.concatenate([mono, np.zeros(fixed_points - n)])
            n = fixed_points

        # Resample to the shared plot resolution using linear interpolation
        indices = np.linspace(0, n - 1, fixed_points)
        mono_plot = np.interp(indices, np.arange(n), mono)

        # Vectorized coordinate computation
        wave_height = (h // 2) - 20  # Leave margin at top/bottom
        xs = (np.arange(fixed_points) / max(1, fixed_points - 1)) * w
        ys = mid_y - (mono_plot * wave_height).astype(int)

        # Interleave x,y coordinates (left/right). Size stays constant via WAVEFORM_PLOT_POINTS.
        coords = np.column_stack((xs, ys)).ravel().tolist()

        # Create line once, then always reuse (coords length never changes)
        if self._wave_id is None:
            self._wave_id = c.create_line(coords, fill="#00ff88", width=1.5, smooth=False)
        else:
            c.coords(self._wave_id, *coords)

        # Reuse center line (only recreate if height changed)
        if self._mid_id is None or self._last_h != h:
            if self._mid_id is not None:
                c.delete(self._mid_id)
            self._mid_id = c.create_line(0, mid_y, w, mid_y, fill="#444444", width=1, dash=(4, 2))
            self._last_h = h
        else:
            # Update center line position (width might have changed)
            c.coords(self._mid_id, 0, mid_y, w, mid_y)


# ----------------------- Presets -----------------------


class audioApp(tk.Tk):
    # Path to user presets file - same directory as this script
    USER_PRESETS_FILE = Path(__file__).parent / "presets.json"

    # Built-in Presets
    PRESETS = {
        "Default": {
        "dur": 60.0,     # longer arc to really show 1→∞→1
        "base": 96.5,     # musical, stable anchor
        "voices": 6,      # enough grain without mush
        "layers": 5,      # max UI depth; distinct recursive motion
        "vdelay": 6.0,    # brief unity before emergence
        "breath": 0.012,  # subtle organic wobble, not seasick
        "width": 0.60,    # bold early spread that collapses
        "bdel": 11.0,     # audible early binaural sway
        "bamt": 0.28,     # present but not gimmicky
        "even": 0.12,     # modest Chebyshev color
        "odd": 0.10,      # keeps timbre alive during collapse
        "comb": 0.12,     # light modal shimmer
        "curve": 2.6      # “hold → tip → return” late collapse
        }
    }

    # Preset defaults - used when user presets are missing keys
    _PRESET_DEFAULTS = {
        "dur": 60.0,     # longer arc to really show 1→∞→1
        "base": 96.5,     # musical, stable anchor
        "voices": 6,      # enough grain without mush
        "layers": 5,      # max UI depth; distinct recursive motion
        "vdelay": 6.0,    # brief unity before emergence
        "breath": 0.012,  # subtle organic wobble, not seasick
        "width": 0.60,    # bold early spread that collapses
        "bdel": 11.0,     # audible early binaural sway
        "bamt": 0.28,     # present but not gimmicky
        "even": 0.12,     # modest Chebyshev color
        "odd": 0.10,      # keeps timbre alive during collapse
        "comb": 0.12,     # light modal shimmer
        "curve": 2.6      # “hold → tip → return” late collapse
    }

    # Valid parameter ranges for clamping (guards against bad JSON values)
    _PRESET_RANGES = {
        "dur": (5, 600),
        "base": (20, 220),
        "voices": (1, 9),
        "layers": (1, 9),
        "vdelay": (0, 30),
        "breath": (0.0, 0.3),
        "width": (0, 1),
        "bdel": (0, 120),
        "bamt": (0, 0.6),
        "even": (0, 0.6),
        "odd": (0, 0.6),
        "comb": (0, 0.5),
        "curve": (1.2, 3.5),
    }
    
    # ----------------------- GUI -----------------------

    # Parameter descriptions
    PARAM_DESCRIPTIONS = {
        "Duration (s)": "Length of the composition in seconds. Longer durations allow for more gradual evolution toward the base tone.",
        "Base f0 (Hz)": "Fundamental frequency - the target pitch for the final tone. Lower values create deeper drones; higher values are brighter.",
        "Voices": "Number of detuned oscillators. More voices create richer initial textures that collapse toward unity.",
        "Layers": "FM/AM recursion depth. More layers create more complex timbral evolution through recursive modulation.",
        "Voice delay (s)": 'Duration of initial unity phase. The base tone plays alone before voices emerge, emphasizing the "one → many → one" philosophy.',
        "Breath rate (Hz)": "Frequency of slow amplitude oscillation. Lower values (0.01-0.05 Hz) create slower, meditative breathing patterns. Higher values (0.1-0.2 Hz) create faster pulsing.",
        "Stereo width": "Spatial decorrelation amount. Creates stereo separation that collapses to mono as the piece progresses toward the base tone.",
        "Binaural Δ (Hz)": "Interaural frequency difference in Hz. Creates binaural beats that fade out as the event collapses. Tip: set Δ ≈ 0.343 × Base f0 to place left/right a tritone apart (ratio ≈ √2).",
        "Binaural amt": "Strength of the binaural beat layer. Higher values make the beat effect more prominent in the early stages.",
        "Even harmonic": "Harmonic enrichment using 2nd-order Chebyshev polynomial (T2). Adds even harmonics to the timbre.",
        "Odd harmonic": "Harmonic enrichment using 3rd-order Chebyshev polynomial (T3). Adds odd harmonics to the timbre.",
        "Comb amount": "Modal resonance strength via feedforward comb filtering. Emphasizes harmonic partials, creating bell-like resonances.",
        "Collapse curve": "Exponential shape of the return envelope. Higher values create faster final collapse; lower values spread the transition more evenly. Values greater than 1.0 creates a non-linear decay that can feel more musically natural than linear fading, but it also means most of the dramatic change happens in the latter portion of the piece, which could make the beginning feel static if not balanced with other evolving elements.",
    }

    # Built-in preset names (automatically derived from PRESETS keys)
    # Used to distinguish built-in presets from user-created ones
    BUILTIN_PRESET_NAMES = frozenset(PRESETS.keys())

    def __init__(self):
        super().__init__()
        # --- App styling (theme, colors, spacing) ---
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        default_font = tkfont.nametofont("TkDefaultFont")
        heading_offset = 16 - default_font["size"]
        note_offset = 10 - default_font["size"]
        body_offset = 14 - default_font["size"]

        self._label_font = default_font.copy()
        self._label_font.configure(size=default_font["size"] + 2)
        self._heading_font = default_font.copy()
        self._heading_font.configure(
            size=default_font["size"] + heading_offset + 2, weight="bold"
        )
        self._note_font = default_font.copy()
        self._note_font.configure(
            size=default_font["size"] + note_offset + 2, weight="bold"
        )
        self._body_font = default_font.copy()
        self._body_font.configure(size=default_font["size"] + body_offset + 2)
        self._status_font = self._label_font

        # Mild contrasted backgrounds
        self._APP_BG = "#ECEFF3"   # window bg
        self._LEFT_BG = "#F6F8FB"  # control panel
        self._RIGHT_BG = "#FFFFFF" # visualization area
        self._SEP_BG = "#D5DADD"   # separator line

        try:
            self.configure(bg=self._APP_BG)
        except Exception:
            pass

        style.configure("Left.TFrame",  background=self._LEFT_BG)
        style.configure("Right.TFrame", background=self._RIGHT_BG)
        # Use white backgrounds for clean contrast, remove grey overlay from entries and labels
        style.configure("App.TLabel",   background=self._APP_BG)
        style.configure("Left.TLabel",  background=self._LEFT_BG, font=self._label_font)
        style.configure("Left.TCheckbutton", background=self._LEFT_BG, font=self._label_font)
        style.configure("Left.TNotebook", background=self._LEFT_BG, borderwidth=0)
        style.configure("Left.TNotebook.Tab", background=self._LEFT_BG, font=self._label_font)
        style.map("Left.TNotebook.Tab", background=[("selected", "#FFFFFF")])
        style.configure("Left.Horizontal.TScale", background=self._LEFT_BG)
        style.configure("TEntry", fieldbackground="#FFFFFF", background="#FFFFFF")
        style.configure("TCombobox", fieldbackground="#FFFFFF", background="#FFFFFF")

        self.title("Recursion/Strange Attractor Inspired Conceptual Audio Synthesizer — Generator + Lissajous")
        self.geometry("980x660")

        self.audio = None  # numpy stereo float
        self.sr = AUDIO_SR
        self._animating = False  # flag to track if animation is running
        self._anim_frame_counter = 0  # counter for skipping slider updates

        # Stable visualization tracking (sample-based, not time-based)
        self._viz_position = 0  # Current playback position in samples
        self._viz_window_samples = int(VIZ_WINDOW_SECONDS * AUDIO_SR)

        # sounddevice streaming state
        self._sd_stream = None
        self._sd_position = 0
        self._sd_pos_seq = 0  # Monotonic sequence number for detecting new callbacks
        self._sd_last_t = 0.0  # Timestamp of last callback (perf_counter)
        self._sd_callback_period = 0.023  # Rolling average callback period (23ms typical)
        self._sd_stop_flag = False
        self._sd_lock = threading.Lock()  # Protects _sd_position, _sd_pos_seq, _sd_last_t

        # Visualization state (UI thread)
        self._viz_last_seq = -1  # Last seen sequence number for extrapolation
        self._viz_last_pos = 0  # Last known position for smooth easing
        self._effective_len = 0        # actual number of samples in self.audio
        self._shutting_down = False
        # Control-rate activity (for ΔΦ meter)
        self.ctrl_progress = None   # control-rate 0..1 timeline (Nc,)
        self.ctrl_activity = None   # smoothed |d ctrl| at control rate (Nc,)

        self.base_f0_note_label = None  # will hold the note name label

        # Load user presets from file
        self._load_user_presets()

        self._build_ui()

        # Register cleanup handler to stop audio on window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._anim_after_id = None

    # ---- UI construction ----
    def _build_ui(self):
        # Left (controls), Separator, Right (visuals)
        ctrl = ttk.Frame(self, style="Left.TFrame", padding=(12, 12))
        ctrl.columnconfigure(1, weight=1)
        ctrl.pack(side="left", fill="y")

        # Subtle vertical separator for visual separation
        sep = tk.Frame(self, width=1, bg=self._SEP_BG, highlightthickness=0)
        sep.pack(side="left", fill="y")

        # Right side: visualization area with both Lissajous and Waveform
        viz_container = ttk.Frame(self, style="Right.TFrame", padding=(12, 12))
        viz_container.pack(side="left", fill="both", expand=True)

        # Lissajous on top
        viz = LissajousPane(viz_container)
        viz.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        self.viz = viz
        # Initialize visual FPS from current slider (capped at 60 by setter)
        try:
            self.viz.set_target_fps(self.var_fps.get())
        except Exception:
            pass

        # Waveform on bottom
        waveform = WaveformPane(viz_container)
        waveform.pack(side="bottom", fill="both", expand=True, padx=6, pady=6)
        self.waveform = waveform

        # Lissajous display options
        self.var_liss_show_axes = tk.BooleanVar(value=False)

        def _on_liss_options_change():
            self.viz.set_display_options(
                show_axes=self.var_liss_show_axes.get(),
                auto_scale=getattr(self.viz, "auto_scale", False),
            )

        # Apply initial state
        _on_liss_options_change()

        # Controls - default to Default preset
        self.var_dur = tk.DoubleVar(value=60.0)
        self.var_sr = tk.IntVar(value=AUDIO_SR)
        self.var_base = tk.DoubleVar(value=96.5)
        self.var_voices = tk.IntVar(value=7)
        self.var_layers = tk.IntVar(value=5)
        self.var_vdelay = tk.DoubleVar(value=6.0)
        self.var_breath = tk.DoubleVar(value=0.012)
        self.var_width = tk.DoubleVar(value=0.65)
        self.var_bdel = tk.DoubleVar(value=11.0)
        self.var_bamt = tk.DoubleVar(value=0.28)
        self.var_even = tk.DoubleVar(value=0.12)
        self.var_odd = tk.DoubleVar(value=0.10)
        self.var_comb = tk.DoubleVar(value=0.12)
        self.var_curve = tk.DoubleVar(value=2.6)
        
        # Visualization toggles
        self.var_show_lissajous = tk.BooleanVar(value=True)
        self.var_show_waveform = tk.BooleanVar(value=True)
        self.var_fps = tk.IntVar(value=60)  # Default 60 FPS
   
        self.var_stroke_width = tk.DoubleVar(value=1)
        self.var_stroke_display = tk.StringVar(value="1 px")

        def add_slider(parent, row, text, var, frm=0.0, to=1.0, res=0.01):
            label = tk.Label(
                parent,
                text=text,
                cursor="hand2",
                fg="#0066cc",
                bg=self._LEFT_BG,
                font=self._label_font,
            )
            label.grid(row=row, column=0, sticky="w")
            label.bind("<Button-1>", lambda e, t=text: self.show_description(t))

            # Command to round values to the specified resolution
            def round_to_resolution(value):
                rounded = round(float(value) / res) * res
                var.set(rounded)

            s = ttk.Scale(
                parent,
                variable=var,
                from_=frm,
                to=to,
                orient="horizontal",
                command=round_to_resolution,
                style="Left.Horizontal.TScale",
            )
            s.grid(row=row, column=1, sticky="ew", padx=6)

            # Editable value label
            value_frame = tk.Frame(parent, bg=self._LEFT_BG)
            value_frame.grid(row=row, column=2)

            value_label = tk.Label(
                value_frame,
                textvariable=var,
                width=7,
                cursor="hand2",
                fg="#0066cc",
                bg=self._LEFT_BG,
                font=self._label_font,
            )
            value_label.pack()

            # Entry widget for editing (hidden by default)
            value_entry = tk.Entry(
                value_frame,
                width=7,
                bg=self._LEFT_BG,
                fg="#000000",
                relief="solid",
                borderwidth=1,
                font=self._label_font,
            )

            def start_edit(event):
                value_label.pack_forget()
                value_entry.delete(0, "end")
                value_entry.insert(0, str(var.get()))
                value_entry.pack()
                value_entry.focus_set()
                value_entry.select_range(0, "end")

            def finish_edit(event=None):
                try:
                    new_val = float(value_entry.get())
                    # Clamp to valid range
                    new_val = max(frm, min(to, new_val))
                    # Round to resolution
                    new_val = round(new_val / res) * res
                    var.set(new_val)
                except ValueError:
                    pass  # Invalid input, keep old value
                value_entry.pack_forget()
                value_label.pack()

            def cancel_edit(event):
                value_entry.pack_forget()
                value_label.pack()

            value_label.bind("<Button-1>", start_edit)
            value_entry.bind("<Return>", finish_edit)
            value_entry.bind("<KP_Enter>", finish_edit)
            value_entry.bind("<FocusOut>", finish_edit)
            value_entry.bind("<Escape>", cancel_edit)

            return s

        ctrl.columnconfigure(1, weight=1)
        r = 0

        # Preset selector
        tk.Label(
            ctrl,
            text="Presets",
            font=self._heading_font,
            anchor="w",
            justify="left",
            bg=self._LEFT_BG,
            fg="#000000",
        ).grid(row=r, column=0, columnspan=3, pady=(0, 6), sticky="w")
        r += 1

        preset_frame = tk.Frame(ctrl, bg=self._LEFT_BG)
        preset_frame.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(0, 32))

        tk.Label(
            preset_frame,
            text="Preset:",
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
        ).pack(side="left", padx=(0, 6))

        self.preset_var = tk.StringVar(value="Default")
        self.preset_menu = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=list(self.PRESETS.keys()),
            state="readonly",
            width=20,
        )
        self.preset_menu.pack(side="left", padx=(0, 6))
        self.preset_menu.bind("<<ComboboxSelected>>", lambda e: self.load_preset())

        ttk.Button(
            preset_frame, text="Save...", command=self.save_current_as_preset
        ).pack(side="left", padx=2)

        ttk.Button(
            preset_frame, text="Reset", command=self.reset_to_minimum
        ).pack(side="left", padx=2)

        r += 1

        tk.Label(
            ctrl,
            text="Core Controls",
            font=self._heading_font,
            anchor="w",
            justify="left",
            bg=self._LEFT_BG,
            fg="#000000",
        ).grid(row=r, column=0, columnspan=3, pady=(0, 6), sticky="w")
        r += 1
        add_slider(ctrl, r, "Duration (s)", self.var_dur, 5, 600, 1)
        r += 1

        # Base f0 with note name display - handled specially
        label = tk.Label(
            ctrl,
            text="Base f0 (Hz)",
            cursor="hand2",
            fg="#0066cc",
            bg=self._LEFT_BG,
            font=self._label_font,
        )
        label.grid(row=r, column=0, sticky="w")
        label.bind("<Button-1>", lambda e: self.show_description("Base f0 (Hz)"))

        def round_base_f0(value):
            rounded = round(float(value) / 0.5) * 0.5
            self.var_base.set(rounded)
            self.update_base_f0_note()

        s = ttk.Scale(
            ctrl,
            variable=self.var_base,
            from_=20,
            to=220,
            orient="horizontal",
            command=round_base_f0,
        )
        s.grid(row=r, column=1, sticky="ew", padx=6)

        # Value frame with editable label and note name (grid so the number can be right‑aligned)
        value_frame = tk.Frame(ctrl, bg=self._LEFT_BG)
        value_frame.grid(row=r, column=2, sticky="e")  # align to the right like other value labels
        value_frame.columnconfigure(0, weight=1)  # numeric column expands to push content to the right
        value_frame.columnconfigure(1, weight=0)

        value_label = tk.Label(
            value_frame,
            textvariable=self.var_base,
            width=7,
            cursor="hand2",
            fg="#0066cc",
            bg=self._LEFT_BG,
            font=self._label_font,
            anchor="e",   # right‑align inside the label
            justify="right",
        )
        value_label.grid(row=0, column=0, sticky="e")

        self.base_f0_note_label = tk.Label(
            value_frame,
            text="",
            fg="#666666",
            font=self._note_font,
            bg=self._LEFT_BG,
        )
        self.base_f0_note_label.grid(row=0, column=1, sticky="w", padx=(4, 0))

        # Entry widget for editing (appears in place of both labels)
        value_entry = tk.Entry(
            value_frame,
            width=7,
            bg=self._LEFT_BG,
            fg="#000000",
            relief="solid",
            borderwidth=1,
            font=self._label_font,
            justify="right",
        )

        def start_edit_base(event):
            # Hide labels and show entry spanning both columns so the caret lines up on the right
            value_label.grid_remove()
            self.base_f0_note_label.grid_remove()
            value_entry.delete(0, "end")
            value_entry.insert(0, str(self.var_base.get()))
            value_entry.grid(row=0, column=0, columnspan=2, sticky="e")
            value_entry.focus_set()
            value_entry.select_range(0, "end")

        def finish_edit_base(event=None):
            try:
                new_val = float(value_entry.get())
                new_val = max(20, min(220, new_val))
                new_val = round(new_val / 0.5) * 0.5
                self.var_base.set(new_val)
                self.update_base_f0_note()
            except ValueError:
                pass
            value_entry.grid_remove()
            value_label.grid(row=0, column=0, sticky="e")
            self.base_f0_note_label.grid(row=0, column=1, sticky="w", padx=(4, 0))

        def cancel_edit_base(event):
            value_entry.grid_remove()
            value_label.grid(row=0, column=0, sticky="e")
            self.base_f0_note_label.grid(row=0, column=1, sticky="w", padx=(4, 0))

        value_label.bind("<Button-1>", start_edit_base)
        value_entry.bind("<Return>", finish_edit_base)
        value_entry.bind("<KP_Enter>", finish_edit_base)
        value_entry.bind("<FocusOut>", finish_edit_base)
        value_entry.bind("<Escape>", cancel_edit_base)

        r += 1
        add_slider(ctrl, r, "Layers (FM/AM depth)", self.var_layers, 1, 5, 1)
        r += 1
        add_slider(ctrl, r, "Voices per layer", self.var_voices, 1, 9, 1)
        r += 1

        # Parameter Notebook for advanced sections
        param_notebook = ttk.Notebook(ctrl, style="Left.TNotebook")
        param_notebook.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 32))
        r += 1

        # Temporal parameters
        temporal_frame = tk.Frame(param_notebook, bg=self._LEFT_BG)
        temporal_frame.columnconfigure(1, weight=1)
        add_slider(temporal_frame, 0, "Voice delay (s)", self.var_vdelay, 0, 30, 0.5)
        add_slider(temporal_frame, 1, "Breath rate (Hz)", self.var_breath, 0.0, 0.3, 0.005)
        add_slider(temporal_frame, 2, "Collapse curve", self.var_curve, 1.2, 3.5, 0.05)
        param_notebook.add(temporal_frame, text="Temporal")

        # Spatial parameters
        spatial_frame = tk.Frame(param_notebook, bg=self._LEFT_BG)
        spatial_frame.columnconfigure(1, weight=1)
        add_slider(spatial_frame, 0, "Stereo width", self.var_width, 0, 1, 0.01)
        add_slider(spatial_frame, 1, "Binaural balance (Hz)", self.var_bdel, 0, 120, 0.1)
        add_slider(spatial_frame, 2, "Binaural amount", self.var_bamt, 0, 0.6, 0.01)
        param_notebook.add(spatial_frame, text="Spatial")

        # Timbral parameters
        timbral_frame = tk.Frame(param_notebook, bg=self._LEFT_BG)
        timbral_frame.columnconfigure(1, weight=1)
        add_slider(timbral_frame, 0, "Even harmonic", self.var_even, 0, 0.6, 0.01)
        add_slider(timbral_frame, 1, "Odd harmonic", self.var_odd, 0, 0.6, 0.01)
        add_slider(timbral_frame, 2, "Comb amount", self.var_comb, 0, 0.5, 0.01)
        param_notebook.add(timbral_frame, text="Timbral")

        # Visualization toggles and remaining controls
        tk.Label(
            ctrl,
            text="Display",
            font=self._heading_font,
            anchor="w",
            justify="left",
            bg=self._LEFT_BG,
            fg="#000000",
        ).grid(
            row=r, column=0, columnspan=3, pady=(8, 6), sticky="w"
        )
        r += 1
        ttk.Checkbutton(
            ctrl,
            text="Lissajous (XY)",
            variable=self.var_show_lissajous,
            command=self.toggle_visualizations,
            style="Left.TCheckbutton",
        ).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        ttk.Checkbutton(
            ctrl,
            text="Waveform",
            variable=self.var_show_waveform,
            command=self.toggle_visualizations,
            style="Left.TCheckbutton",
        ).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1

        # Toggle for Lissajous frame & labels
        ttk.Checkbutton(
            ctrl,
            text="Show Lissajous frame & labels",
            variable=self.var_liss_show_axes,
            command=_on_liss_options_change,
            style="Left.TCheckbutton",
        ).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1

        # Stroke width slider for Lissajous lines
        tk.Label(
            ctrl,
            text="Lissajous stroke (px):",
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
        ).grid(row=r, column=0, sticky="w", pady=(8, 0))

        def _on_stroke_width_change(value):
            try:
                val = float(value)
            except (TypeError, ValueError):
                return
            val = max(0.4, min(2.0, val))
            self.var_stroke_display.set(f"{val:.1f} px")
            self.viz.set_stroke_width(val)

        stroke_slider = ttk.Scale(
            ctrl,
            variable=self.var_stroke_width,
            from_=0.4,
            to=2.0,
            orient="horizontal",
            command=_on_stroke_width_change,
            style="Left.Horizontal.TScale",
        )
        stroke_slider.grid(row=r, column=1, sticky="ew", padx=6, pady=(8, 0))

        tk.Label(
            ctrl,
            textvariable=self.var_stroke_display,
            width=7,
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
        ).grid(row=r, column=2, pady=(8, 0))
        r += 1

        _on_stroke_width_change(self.var_stroke_width.get())

        # FPS slider
        tk.Label(
            ctrl,
            text="Animation FPS:",
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
        ).grid(row=r, column=0, sticky="w", pady=(8, 0))

        def round_fps(value):
            # Round to nearest 5
            rounded = round(float(value) / 5) * 5
            self.var_fps.set(int(rounded))

        fps_slider = ttk.Scale(
            ctrl,
            variable=self.var_fps,
            from_=20,
            to=60,
            orient="horizontal",
            command=round_fps,
            style="Left.Horizontal.TScale",
        )
        fps_slider.grid(row=r, column=1, sticky="ew", padx=6, pady=(8, 0))

        tk.Label(
            ctrl,
            textvariable=self.var_fps,
            width=7,
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
        ).grid(row=r, column=2, pady=(8, 0))
        r += 1

        # Buttons
        btns = tk.Frame(ctrl, bg=self._LEFT_BG)
        btns.grid(row=r, column=0, columnspan=3, pady=8, sticky="ew")
        r += 1
        ttk.Button(btns, text="Generate", command=self.on_generate).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Save As…", command=self.on_save_as).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Load WAV", command=self.on_load_wav).pack(
        side="left", padx=4
        )
        ttk.Button(btns, text="Play", command=self.on_play).pack(side="left", padx=4)
        ttk.Button(btns, text="Stop", command=self.on_stop).pack(side="left", padx=4)
 

        # Time slider for scrubbing/preview
        self.var_time = tk.DoubleVar(value=0.0)
        tk.Label(
            ctrl,
            text="Start (s)",
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
        ).grid(row=r, column=0, sticky="w")
        self.s_time = ttk.Scale(
            ctrl,
            variable=self.var_time,
            from_=0.0,
            to=self.var_dur.get(),
            orient="horizontal",
            command=self._on_time_slider_change,
            style="Left.Horizontal.TScale",
        )
        self.s_time.grid(row=r, column=1, columnspan=2, sticky="ew", padx=6)
        try:
            self.var_dur.trace_add("write", lambda *args: self._refresh_time_slider_limit())
        except Exception:
            pass
        self._refresh_time_slider_limit()
        r += 1

        # Tiny ΔΦ activity meter (smoothed |d ctrl|) just below the time slider
        tk.Label(
            ctrl,
            text="ΔΦ",
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
            width=3,
            anchor="w",
        ).grid(row=r, column=0, sticky="w")
        self.activity_canvas = tk.Canvas(
            ctrl,
            height=22,
            bg="#F0F3F7",
            highlightthickness=1,
            highlightbackground=self._SEP_BG,
        )
        self.activity_canvas.grid(row=r, column=1, columnspan=2, sticky="ew", padx=6)
        self.activity_canvas.bind("<Configure>", lambda e: self._draw_activity_meter())
        r += 1

        # Value display below the slider
        value_display_frame = tk.Frame(ctrl, bg=self._LEFT_BG)
        value_display_frame.grid(row=r, column=0, columnspan=3, sticky="w")

        tk.Label(
            value_display_frame,
            text="Time(s):",
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
        ).pack(side="left", padx=(0, 6))

        tk.Label(
            value_display_frame,
            textvariable=self.var_time,
            width=20,
            bg=self._LEFT_BG,
            font=self._label_font,
            fg="#000000",
        ).pack(side="left")

        tk.Label(
            value_display_frame,
            text="",
            bg=self._LEFT_BG,
            font=self._label_font,
        ).pack(side="left", padx=(2, 0))
        r += 1

        ctrl.rowconfigure(r, weight=1)
        r += 1

        bottom_frame = tk.Frame(ctrl, bg=self._LEFT_BG)
        bottom_frame.grid(row=r, column=0, columnspan=3, sticky="sew", pady=(0, 16))
        bottom_frame.columnconfigure(0, weight=1)

        info_label = tk.Label(
            bottom_frame,
            text="Parameter Info",
            font=self._heading_font,
            anchor="w",
            justify="left",
            bg=self._LEFT_BG,
            fg="#000000",
        )
        info_label.pack(anchor="w", pady=(0, 8))

        self.desc_text = tk.Text(
            bottom_frame,
            height=5,
            wrap="word",
            bg=self._LEFT_BG,
            fg="#000000",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=8,
            font=self._body_font,
        )
        self.desc_text.pack(fill="both", expand=True, pady=(0, 8))
        self.desc_text.insert(
            "1.0", "Click any parameter label above to see its description."
        )
        self.desc_text.config(state="disabled")  # Make it read-only

        self.status = tk.Label(
            bottom_frame,
            text="Ready",
            bg=self._LEFT_BG,
            font=self._status_font,
            fg="#000000",
        )
        self.status.pack(anchor="w")

        # Draw initial note name
        self.after(100, self.update_base_f0_note)

        # Update preset menu with any loaded user presets
        self.preset_menu["values"] = list(self.PRESETS.keys())

    # ---- Preset Management ----
    def _load_user_presets(self):
        """Load user presets from JSON file on disk with validation."""
        if not self.USER_PRESETS_FILE.exists():
            return

        try:
            with open(self.USER_PRESETS_FILE, "r") as f:
                user_presets = json.load(f)

            # Validate that the file contains a dictionary
            if not isinstance(user_presets, dict):
                raise ValueError("Presets file must contain a JSON object")

            # Validate and sanitize each user preset
            clean = {}
            for name, preset in user_presets.items():
                if isinstance(preset, dict):
                    # Validate and sanitize (silently fixes issues during load)
                    sanitized, _ = self._validate_preset(preset)
                    clean[name] = sanitized

            # Merge validated presets with built-ins
            self.PRESETS.update(clean)

            # Refresh combobox values if it exists
            try:
                if hasattr(self, 'preset_menu'):
                    self.preset_menu.configure(values=list(self.PRESETS.keys()))
            except Exception:
                pass

        except Exception as e:
            # Show warning for file-level errors
            try:
                messagebox.showwarning(
                    "Preset Load Error",
                    f"Could not load user presets:\n{e}"
                )
            except Exception:
                print(f"Warning: Could not load user presets: {e}")

    def _save_user_presets(self):
        """Save all non-built-in presets to JSON file."""
        # Extract only user-created presets
        user_presets = {
            name: params
            for name, params in self.PRESETS.items()
            if name not in self.BUILTIN_PRESET_NAMES
        }

        try:
            with open(self.USER_PRESETS_FILE, "w") as f:
                json.dump(user_presets, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save presets: {e}")

    def _get_current_preset_dict(self):
        """Get current parameter values as a preset dictionary."""
        return {
            "dur": self.var_dur.get(),
            "base": self.var_base.get(),
            "voices": int(self.var_voices.get()),
            "layers": int(self.var_layers.get()),
            "vdelay": self.var_vdelay.get(),
            "breath": self.var_breath.get(),
            "width": self.var_width.get(),
            "bdel": self.var_bdel.get(),
            "bamt": self.var_bamt.get(),
            "even": self.var_even.get(),
            "odd": self.var_odd.get(),
            "comb": self.var_comb.get(),
            "curve": self.var_curve.get(),
        }

    def save_current_as_preset(self):
        """Save current parameter values as a new preset."""
        # Create dialog to get preset name
        dialog = tk.Toplevel(self)
        dialog.title("Save Preset")
        dialog.geometry("350x120")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="Preset name:").pack(pady=(10, 5))

        name_var = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=name_var, width=40)
        entry.pack(pady=5)
        entry.focus_set()

        def do_save():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Invalid Name", "Preset name cannot be empty.")
                return

            # Check if overwriting built-in
            if name in self.BUILTIN_PRESET_NAMES:
                if not messagebox.askyesno(
                    "Overwrite Built-in?",
                    f'"{name}" is a built-in preset. This will only affect this session.\nContinue?',
                ):
                    return

            # Save to presets dict
            self.PRESETS[name] = self._get_current_preset_dict()

            # Update preset dropdown
            self.preset_menu["values"] = list(self.PRESETS.keys())
            self.preset_var.set(name)

            # Save to disk (only non-built-in presets)
            self._save_user_presets()

            self.status.config(text=f"Saved preset: {name}")
            dialog.destroy()

        def cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Save", command=do_save).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(
            side="left", padx=5
        )

        # Bind Enter key
        entry.bind("<Return>", lambda e: do_save())
        entry.bind("<Escape>", lambda e: cancel())

    def reset_to_minimum(self):
        """Reset all parameters to their off or minimum values."""
        # Set all parameters to off (0) or minimum values
        self.var_dur.set(5)          # Minimum duration for quick testing
        self.var_base.set(20)         # Lowest frequency
        self.var_voices.set(1)        # Minimum voices
        self.var_layers.set(1)        # Minimum layers
        self.var_vdelay.set(0)        # Off
        self.var_breath.set(0.0)      # Off
        self.var_width.set(0)         # Off (mono)
        self.var_bdel.set(0)          # Off
        self.var_bamt.set(0)          # Off
        self.var_even.set(0)          # Off
        self.var_odd.set(0)           # Off
        self.var_comb.set(0)          # Off
        self.var_curve.set(1.2)       # Minimum collapse curve

        # Update visualizations
        self.update_base_f0_note()

        # Keep the time slider in range if audio already exists
        try:
            self._refresh_time_slider_limit()
        except Exception:
            pass

        # Clear preset selection to indicate custom state
        self.preset_var.set("")

        # Update status
        self.status.config(text="Reset to minimum values")

    # ---- Preset Validation Helpers ----
    def _validate_preset(self, preset: dict):
        """
        Validate and sanitize a preset dictionary.

        Args:
            preset: Raw preset dictionary (may have missing keys, wrong types, or out-of-range values)

        Returns:
            tuple: (sanitized_dict, list_of_issues)
                - sanitized_dict: Clean preset with all keys present, valid types, and clamped values
                - list_of_issues: List of human-readable strings describing corrections made
        """
        issues = []
        out = {}

        for k, default in self._PRESET_DEFAULTS.items():
            raw = preset.get(k, default)
            try:
                val = float(raw)
                lo, hi = self._PRESET_RANGES[k]
                if val < lo or val > hi:
                    issues.append(f"{k}={raw} clamped to [{lo}, {hi}]")
                val = float(np.clip(val, lo, hi))

                # Integer variables stay integers
                if k in ("voices", "layers"):
                    val = int(round(val))
            except (TypeError, ValueError):
                issues.append(f"{k}={raw} invalid → using default {default}")
                val = default

            out[k] = val

        return out, issues

    def _preset_var_map(self):
        """
        Return mapping from preset keys to Tk variables.

        Returns:
            dict: Mapping of preset key strings to their corresponding Tk variables
        """
        return {
            "dur": self.var_dur,
            "base": self.var_base,
            "voices": self.var_voices,
            "layers": self.var_layers,
            "vdelay": self.var_vdelay,
            "breath": self.var_breath,
            "width": self.var_width,
            "bdel": self.var_bdel,
            "bamt": self.var_bamt,
            "even": self.var_even,
            "odd": self.var_odd,
            "comb": self.var_comb,
            "curve": self.var_curve,
        }

    # ---- Actions ----
    def load_preset(self):
        """Load selected preset values into all parameters with robust validation."""
        name = self.preset_var.get()
        preset = self.PRESETS.get(name)

        # Safety: fall back to defaults if preset is missing or malformed
        if not isinstance(preset, dict):
            preset = self._PRESET_DEFAULTS.copy()

        # Policy: Trust built-in presets completely, validate user presets
        # Built-ins are curated and may intentionally exceed UI slider ranges
        # (e.g., Perfect Fifth has curve=20.05, many presets have bamt=0.90-1.0)
        is_builtin = name in self.BUILTIN_PRESET_NAMES

        if is_builtin:
            # Trust built-ins verbatim - use values as-is
            sanitized = preset
            issues = []
        else:
            # Validate and sanitize user presets (handles missing keys, wrong types, out-of-range values)
            sanitized, issues = self._validate_preset(preset)

        # Apply values to UI variables
        for k, var in self._preset_var_map().items():
            value = sanitized.get(k, self._PRESET_DEFAULTS[k])
            var.set(value)

        # Keep the time slider in range if audio already exists
        try:
            self._refresh_time_slider_limit()
        except Exception:
            pass

        # Update visualizations
        self.update_base_f0_note()

        # Warn user if their preset had issues (only for user presets with problems)
        if issues and not is_builtin:
            msg = "Some values in this preset were corrected:\n• " + "\n• ".join(issues)
            try:
                messagebox.showwarning("Preset Corrected", msg)
            except Exception:
                print(msg)

        # Status update
        try:
            self.status.config(text=f"Loaded preset: {name}")
        except Exception:
            pass

    def update_base_f0_note(self):
        """Update the note name label for base f0."""
        if self.base_f0_note_label is not None:
            freq = self.var_base.get()
            note_name = freq_to_note_name(freq)
            self.base_f0_note_label.config(text=f"({note_name})")

    def show_description(self, param_name):
        """Display description for the clicked parameter."""
        description = self.PARAM_DESCRIPTIONS.get(
            param_name, "No description available."
        )
        self.desc_text.config(state="normal")
        self.desc_text.delete("1.0", "end")
        self.desc_text.insert("1.0", f"{param_name}:\n{description}")
        self.desc_text.config(state="disabled")

    def _draw_activity_meter(self):
        """Draw the ΔΦ (smoothed |d ctrl|) sparkline and current time marker."""
        c = getattr(self, "activity_canvas", None)
        if c is None:
            return
        try:
            w = max(10, int(c.winfo_width()))
            h = max(6, int(c.winfo_height()))
        except Exception:
            return
        c.delete("all")
        if self.ctrl_activity is None or self.ctrl_progress is None or self._effective_len <= 0:
            return
        act = np.asarray(self.ctrl_activity, dtype=float)
        if act.size < 2:
            return
        if np.all(np.isfinite(act)):
            vmax = float(np.percentile(act, 95))
        else:
            vmax = 0.0
        if not np.isfinite(vmax) or vmax <= 1e-9:
            vmax = 1.0
        y01 = np.clip(act / vmax, 0.0, 1.0)
        idx = np.linspace(0, y01.size - 1, w)
        y_rs = np.interp(idx, np.arange(y01.size), y01)
        pad = 2
        ys = (h - pad) - (h - 2 * pad) * y_rs
        xs = np.arange(w)
        coords = np.column_stack((xs, ys)).ravel().tolist()
        c.create_line(*coords, fill="#5B6DFF", width=1)
        try:
            t = float(self.var_time.get())
        except Exception:
            t = 0.0
        total_t = max(1e-9, self._effective_len / float(self.sr))
        x_mark = int(np.clip((t / total_t) * (w - 1), 0, w - 1))
        c.create_line(x_mark, 0, x_mark, h, fill="#D02090", width=1)

    def _refresh_time_slider_limit(self):
        slider = getattr(self, "s_time", None)
        if slider is None:
            return
        try:
            if self.audio is not None and len(self.audio) > 0 and self.sr > 0:
                max_time = len(self.audio) / float(self.sr)
            else:
                max_time = float(self.var_dur.get())
        except Exception:
            return
        try:
            slider.configure(to=max(0.0, max_time))
        except Exception:
            pass

    def toggle_visualizations(self):
        """Show/hide visualizations based on toggle state."""
        show_liss = self.var_show_lissajous.get()
        show_wave = self.var_show_waveform.get()

        if show_liss and show_wave:
            # Both visible
            self.viz.pack(side="top", fill="both", expand=True, padx=6, pady=6)
            self.waveform.pack(side="bottom", fill="both", expand=True, padx=6, pady=6)
        elif show_liss and not show_wave:
            # Only Lissajous
            self.waveform.pack_forget()
            self.viz.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        elif not show_liss and show_wave:
            # Only Waveform
            self.viz.pack_forget()
            self.waveform.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        else:
            # Neither visible - show a message
            self.viz.pack_forget()
            self.waveform.pack_forget()

    def on_generate(self):
        # Hard cap: 600 minutes -> 36,000 seconds
        MAX_DUR_SECONDS = 600 * 60

        # Read UI duration and clamp it so the UI stays honest
        dur = float(self.var_dur.get())
        if dur > MAX_DUR_SECONDS:
            dur = MAX_DUR_SECONDS
            self.var_dur.set(dur)

        self.status.config(text="Generating…")
        self.update_idletasks()

        # Generate fresh audio with the clamped duration
        L, R, sr, ctrl_prog, dphi = generate_app(
            dur=dur,
            sr=self.var_sr.get(),
            base_f0=self.var_base.get(),
            voices=int(self.var_voices.get()),
            layers=int(self.var_layers.get()),
            stereo_width=self.var_width.get(),
            binaural_delta_hz0=self.var_bdel.get(),
            binaural_amount=self.var_bamt.get(),
            overtone_power=1.3,
            harmonic_even=self.var_even.get(),
            harmonic_odd=self.var_odd.get(),
            comb_amount=self.var_comb.get(),
            collapse_curve=self.var_curve.get(),
            voice_delay=self.var_vdelay.get(),
            breath_rate=self.var_breath.get(),
        )

        # ---- cleanup old large buffers / visuals so GC can drop them ----
        if getattr(self, "audio", None) is not None:
            self.audio = None
        if hasattr(self, "waveform"):
            self.waveform.show_frame(None)
        if hasattr(self, "viz"):
            self.viz.show_frame(None)
        # ---------------------------------------------------------------

        # Store new audio
        self.sr = sr
        self.audio = np.stack([L, R], axis=1)
        self._effective_len = len(self.audio)

        # Store ΔΦ series for meter
        self.ctrl_progress = np.asarray(ctrl_prog, dtype=float)
        self.ctrl_activity = np.asarray(dphi, dtype=float)

        # Reset scrubber and UI
        self.var_time.set(0.0)
        try:
            self._refresh_time_slider_limit()
        except Exception:
            pass
        try:
            self._draw_activity_meter()
        except Exception:
            pass

        self.status.config(
            text=f"Generated {dur:.1f}s at {sr} Hz • Voices {int(self.var_voices.get())}"
        )
        self.on_show_frame()

    def on_save_as(self):
        if self.audio is None:
            messagebox.showinfo("No audio", "Generate audio first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile="exported_audio.wav",
        )
        if not path:
            return
        write_wav_stereo(path, self.audio[:, 0], self.audio[:, 1], self.sr)
        self.status.config(text=f"Saved to {path}")
        
    def on_load_wav(self):
        """Load an external WAV file for playback and visualization"""
        path = filedialog.askopenfilename(
            title="Load WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            audio_float = None
            n_channels = 0

            # Try scipy.io.wavfile first (handles float WAVs correctly) if available
            try:
                from scipy.io import wavfile
            except ImportError:
                wavfile = None

            if wavfile is not None:
                try:
                    sr, audio_data = wavfile.read(path)

                    # Convert to float32 normalized to [-1, 1]
                    if audio_data.dtype in (np.float32, np.float64):
                        # Already float, just ensure float32
                        audio_float = audio_data.astype(np.float32)
                    elif audio_data.dtype == np.int16:
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_float = audio_data.astype(np.float32) / 2147483648.0
                    elif audio_data.dtype == np.uint8:
                        audio_float = (audio_data.astype(np.float32) - 128.0) / 128.0
                    else:
                        raise ValueError(f"Unsupported dtype: {audio_data.dtype}")

                    n_channels = 1 if audio_float.ndim == 1 else audio_float.shape[1]
                    print(
                        "Loaded with scipy: "
                        f"sr={sr}, original_dtype={audio_data.dtype}, "
                        f"converted_dtype={audio_float.dtype}, channels={n_channels}"
                    )
                    print(
                        f"Original data range: min={np.min(audio_data)}, max={np.max(audio_data)}"
                    )
                    print(
                        f"Converted data range: min={np.min(audio_float)}, max={np.max(audio_float)}"
                    )
                except Exception as exc:
                    audio_float = None
                    print(
                        f"scipy.io.wavfile read failed ({exc}); "
                        "falling back to wave module (may not handle float WAVs correctly)"
                    )
            else:
                print(
                    "scipy not available, using wave module "
                    "(may not handle float WAVs correctly)"
                )

            # Fallback to the built-in wave module when SciPy is unavailable or failed
            if audio_float is None:
                with wave.open(path, "rb") as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    sample_width = wf.getsampwidth()

                    # Read raw audio data
                    raw_data = wf.readframes(n_frames)

                # Convert to numpy array
                if sample_width == 2:  # 16-bit
                    audio_int = np.frombuffer(raw_data, dtype=np.int16)
                    audio_float = audio_int.astype(np.float32) / 32768.0
                elif sample_width == 3:  # 24-bit
                    # Convert 24-bit to int32 then to float
                    audio_bytes = np.frombuffer(raw_data, dtype=np.uint8)
                    audio_int = np.zeros(len(audio_bytes) // 3, dtype=np.int32)
                    for i in range(len(audio_int)):
                        audio_int[i] = (
                            audio_bytes[i * 3]
                            | (audio_bytes[i * 3 + 1] << 8)
                            | (audio_bytes[i * 3 + 2] << 16)
                        )
                        # Sign extend if negative
                        if audio_int[i] & 0x800000:
                            audio_int[i] |= 0xFF000000
                    audio_float = audio_int.astype(np.float32) / 8388608.0
                elif sample_width == 4:  # 32-bit int
                    audio_int = np.frombuffer(raw_data, dtype=np.int32)
                    audio_float = audio_int.astype(np.float32) / 2147483648.0
                else:
                    messagebox.showerror(
                        "Unsupported format",
                        f"Unsupported sample width: {sample_width} bytes",
                    )
                    return

            # Handle mono/stereo
            if n_channels == 1:
                # Mono: duplicate to stereo
                mono = audio_float
                audio = np.column_stack([mono, mono])
            elif n_channels == 2:
                # Stereo: reshape
                audio = audio_float.reshape(-1, 2)
            else:
                # More than 2 channels: take first 2
                audio_temp = audio_float.reshape(-1, n_channels)
                audio = audio_temp[:, :2]

            # Ensure array is C-contiguous for sounddevice
            audio = np.ascontiguousarray(audio, dtype=np.float32)

            # Debug: Check audio stats
            max_amp = np.max(np.abs(audio))
            mean_amp = np.mean(np.abs(audio))
            # Check for NaN or Inf
            has_nan = np.any(np.isnan(audio))
            has_inf = np.any(np.isinf(audio))
            # Sample some values
            sample_vals = audio[:100, 0] if len(audio) > 100 else audio[:, 0]
            print(f"Loaded audio stats: shape={audio.shape}, dtype={audio.dtype}, max_amplitude={max_amp:.6f}, mean={mean_amp:.6f}, has_nan={has_nan}, has_inf={has_inf}, contiguous={audio.flags['C_CONTIGUOUS']}")
            print(f"First 10 samples (L channel): {sample_vals[:10]}")

            # Store loaded audio
            self.audio = audio
            self.sr = sr

            # Update time slider range
            dur = len(audio) / sr
            self._refresh_time_slider_limit()
            self.var_time.set(0.0)

            # Update status
            filename = os.path.basename(path)
            final_max = np.max(np.abs(audio))
            self.status.config(
                text=f"Loaded {filename} • {dur:.1f}s at {sr} Hz • {n_channels} ch • max {final_max:.3f}"
            )

            # Update visualizations
            self.on_show_frame()

        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load WAV file:\n{e}")
            return

    def _sd_callback(self, outdata, frames, time_info, status):
        """sounddevice callback - streams audio from self.audio

        WARNING: This runs in a separate audio thread, not the main thread.
        We ONLY update _sd_position, _sd_pos_seq, _sd_last_t here - no GUI operations allowed.
        """
        import time

        if self._sd_stop_flag or self.audio is None:
            raise sd.CallbackStop

        end_pos = self._sd_position + frames

        if end_pos > len(self.audio):
            # Reached end - fill what we can, pad rest with zeros
            remaining = len(self.audio) - self._sd_position
            if remaining > 0:
                outdata[:remaining] = self.audio[self._sd_position:, :]
            if remaining < frames:
                outdata[remaining:] = 0

            # Atomically update position to end, increment sequence, update timestamp
            with self._sd_lock:
                self._sd_position = len(self.audio)
                self._sd_pos_seq += 1
                self._sd_last_t = time.perf_counter()
            raise sd.CallbackStop

        outdata[:] = self.audio[self._sd_position:end_pos, :]

        # Thread-safe position update (audio thread → main thread)
        # Atomically update position, sequence number, timestamp, and period
        with self._sd_lock:
            now = time.perf_counter()

            # Track callback period for adaptive extrapolation cap
            if self._sd_last_t > 0:
                period = now - self._sd_last_t
                # Exponential moving average: α=0.2 for smoothing
                self._sd_callback_period = 0.8 * self._sd_callback_period + 0.2 * period

            self._sd_last_t = now
            self._sd_position = end_pos
            self._sd_pos_seq += 1

    def _play_via_sounddevice(self, start_seconds=0.0):
        """Play audio using sounddevice streaming (zero disk I/O, minimal RAM)"""
        import time

        if self.audio is None or sd is None:
            return None
        try:
            with self._sd_lock:
                self._sd_position = int(start_seconds * self.sr)
                self._sd_pos_seq = 0  # Reset sequence on new playback
                self._sd_last_t = time.perf_counter()  # Initialize timestamp
            self._sd_stop_flag = False
            self._viz_last_seq = -1  # Reset UI sequence tracker
            self._sd_stream = sd.OutputStream(
                samplerate=self.sr,
                channels=2,
                callback=self._sd_callback,
                dtype='float32',
                blocksize=2048,  # Give UI/MPL some headroom (46ms @ 44.1kHz)
                latency='low'
            )
            self._sd_stream.start()
            return self._sd_stream
        except Exception:
            return None

    def _stop_player(self):
        self._sd_stop_flag = True

        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

    def on_play(self):
        if self.audio is None:
            messagebox.showinfo("No audio", "Generate audio first.")
            return
        start_s = float(self.var_time.get())
        self._animating = False
        self._cancel_animation_timer()
        self._stop_player()
        if self.audio is not None and len(self.audio) > 0:
            max_start = max(0.0, (len(self.audio) - 1) / self.sr)
            start_s = min(start_s, max_start)
        # Try streaming playback via sounddevice
        stream = None
        if sd is not None:
            stream = self._play_via_sounddevice(start_s)

        if stream is None:
            self.status.config(
                text="Playback unavailable — install sounddevice"
            )
            return

        self.status.config(text=f"Playing (sounddevice) from {start_s:.2f}s…")

        # Start animation synchronized with audio
        self._animating = True
        self._anim_frame_counter = 0  # reset frame counter

        # Initialize stable visualization position
        self._viz_position = int(start_s * self.sr)
        self._viz_window_samples = int(VIZ_WINDOW_SECONDS * self.sr)

        self._animate_step()

    def on_stop(self):
        self._animating = False  # stop the animation loop
        self._cancel_animation_timer()
        self._stop_player()
        self.status.config(text="Stopped")
        # release visuals so they don't keep old frames alive
        if hasattr(self, "waveform"):
            self.waveform.show_frame(None)
        if hasattr(self, "viz"):
            self.viz.show_frame(None)

    def _on_closing(self):
        """Gracefully stop playback, cancel animations, and close the window without crashes."""
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        # Cancel animation timer
        try:
            if getattr(self, "_anim_after_id", None) is not None:
                self.after_cancel(self._anim_after_id)
                self._anim_after_id = None
        except Exception:
            pass
        # Stop audio backends
        try:
            self.on_stop()
        except Exception:
            pass
        # Destroy window last
        try:
            self.destroy()
        except Exception:
            pass

    def _on_time_slider_change(self, value):
        """
        Callback for the time slider - receives the slider value as a string.

        Resets extrapolation state so animation doesn't project from stale timing.
        """
        import time

        # Reset extrapolation when user scrubs during playback
        if self._animating and self._sd_stream is not None:
            self._viz_last_seq = -1  # Force fresh data on next frame
            with self._sd_lock:
                # Re-seed timestamp so next extrapolation starts from now
                self._sd_last_t = time.perf_counter()
                # Update position to match slider
                new_pos = int(float(value) * self.sr)
                self._sd_position = min(new_pos, len(self.audio) if self.audio is not None else 0)

        self.on_show_frame()
        try:
            self._draw_activity_meter()
        except Exception:
            pass

    def on_show_frame(self):
        if (
            self.audio is None
            or len(self.audio) == 0
            or not self.viz.winfo_exists()
            or not self.waveform.winfo_exists()
        ):
            return
        i0 = int(float(self.var_time.get()) * self.sr)
        i0 = max(0, min(i0, len(self.audio) - 1))
        # Calculate window size based on current FPS setting
        seconds_per_frame = 1.0 / max(1, self.var_fps.get())
        win = max(1, int(seconds_per_frame * self.sr))
        i1 = min(len(self.audio), i0 + win)
        self.viz.show_frame(self.audio[i0:i1])
        self.waveform.show_frame(self.audio[i0:i1])

    def _animate_step(self):
        """
        Animation step with smooth extrapolation between audio callbacks.

        Uses sequence numbers and timestamps to detect when new audio data has arrived,
        and extrapolates position between callbacks for silky-smooth visual updates.
        """
        import time

        # Early bailout guards - avoid touching widgets after stop
        self._anim_after_id = None
        if not self._animating or self.audio is None:
            return
        if getattr(self, "_shutting_down", False):
            return
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        # Stop visuals when buffer is fully shown
        if self.audio is not None and self._viz_position >= len(self.audio):
            self._animating = False
            return

        # If using sounddevice, sync to its position with hardened extrapolation
        if self._sd_stream is not None and getattr(self._sd_stream, "active", False):
            # Atomically read position, sequence, timestamp, and callback period
            now = time.perf_counter()
            with self._sd_lock:
                pos = self._sd_position
                seq = self._sd_pos_seq
                t_last = self._sd_last_t
                callback_period = self._sd_callback_period

            # Extrapolate only if no new callback arrived since last frame
            if seq == self._viz_last_seq:
                # Same sequence - extrapolate forward using wall-clock time
                elapsed = now - t_last

                # Cap extrapolation to prevent runaway drift during stalls
                # Allow up to 2× the average callback period or 150ms max
                elapsed_cap = min(elapsed, max(2.0 * callback_period, 0.150))
                pos_est = pos + int(elapsed_cap * self.sr)
            else:
                # New callback arrived - use fresh position
                pos_est = pos

                # Smooth large corrections to avoid visible snaps
                delta = pos_est - self._viz_last_pos
                if abs(delta) > int(0.020 * self.sr):  # >20ms correction
                    # Ease toward the target over one frame (half-step)
                    pos_est = self._viz_last_pos + int(0.5 * delta)

            # Remember this sequence and position for next frame
            self._viz_last_seq = seq
            self._viz_last_pos = pos_est

            # Clamp to audio length
            self._viz_position = min(pos_est, len(self.audio))

        # Stop animation cleanly at end-of-buffer
        if self._viz_position >= len(self.audio):
            # Check if stream is truly done (not just extrapolated past end)
            stream_active = (self._sd_stream is not None and
                           getattr(self._sd_stream, "active", False))
            if not stream_active:
                self._animating = False
                self.status.config(text="Playback finished")
                return

        # Check if we've reached the end
        if self._viz_position >= len(self.audio):
            self._animating = False
            self.status.config(text="Playback finished")
            return

        # Fixed window size (constant samples, not time-based)
        i0 = self._viz_position
        i1 = min(len(self.audio), i0 + self._viz_window_samples)

        # Pad with zeros if we're near the end to maintain constant point count
        frame_data = self.audio[i0:i1]
        if len(frame_data) < self._viz_window_samples:
            # Pad with zeros to maintain constant size
            pad_size = self._viz_window_samples - len(frame_data)
            frame_data = np.vstack([frame_data, np.zeros((pad_size, 2))])

        if not (self.viz.winfo_exists() and self.waveform.winfo_exists()):
            return

        self.viz.show_frame(frame_data)
        self.waveform.show_frame(frame_data)

        # Trigger throttled MPL draw for Lissajous (at UI loop rate, not audio rate)
        if hasattr(self, "viz") and self.viz is not None:
            try:
                self.viz._draw_cached_frame()
            except Exception:
                pass

        # Update slider (only every 3rd frame to reduce overhead)
        self._anim_frame_counter += 1
        if self._anim_frame_counter % 3 == 0:
            self.var_time.set(self._viz_position / self.sr)
            try:
                self._draw_activity_meter()
            except Exception:
                pass

        # Advance by fixed hop (samples per frame)
        # hop = sample_rate / FPS
        hop_samples = int(self.sr / max(1, self.var_fps.get()))
        self._viz_position += hop_samples

        # Schedule next frame based on FPS setting
        frame_interval_ms = int((1000.0 / max(1, self.var_fps.get())))
        self._anim_after_id = self.after(frame_interval_ms, self._animate_step)

    def _cancel_animation_timer(self):
        if self._anim_after_id is not None:
            try:
                self.after_cancel(self._anim_after_id)
            except Exception:
                pass
            self._anim_after_id = None


if __name__ == "__main__":
    app = audioApp()
    app.mainloop()
