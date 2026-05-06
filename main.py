import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.animation as animation
import time


def estimate_bpm(y, sr):

    # Break down sample into percussives for accurate beat detetction
    _, y_percussive = librosa.effects.hpss(y)

    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

    # Tempo Calculations
    try:
        from librosa.feature.rhythm import tempo as tempo_func
        tempos = tempo_func(onset_envelope=onset_env, sr=sr, aggregate=None)
    except Exception:
        tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None) # both functions inside error handling as sometimes the other function is not recognised
    bpm = np.median(tempos)

    # Variable bounds for BPM (extreme values)
    if bpm < 35:
        bpm *= 2
    elif bpm > 180:
        bpm /= 2

    return bpm, onset_env


def detect_note_events(y, sr): 
    # weighted pitch modal selection (to reduce lght innacuracies)
    # merge of fragmented segments (needs adjustment still)

    y_harmonic, _ = librosa.effects.hpss(y)
    onset_frames = librosa.onset.onset_detect(y=y_harmonic, sr=sr)
    raw_segments = []
   
    # Weighted Pitch Section
    for i in range(len(onset_frames)):
        start_frame = onset_frames[i]
        end_frame = onset_frames[i + 1] if i + 1 < len(onset_frames) else None

        start_sample = librosa.frames_to_samples(start_frame)
        end_sample = librosa.frames_to_samples(end_frame) if end_frame else len(y)
        segment = y_harmonic[start_sample:end_sample]

        # Minimum segment length (reduces small scale fragmentation)
        if len(segment) < sr * 0.1:
            continue

        # Key note bondaries
        f0, voiced_flag, voiced_prob = librosa.pyin(
            segment,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )

        valid = ~np.isnan(f0)

        f0 = f0[valid]
        voiced_prob = voiced_prob[valid]

        if len(f0) == 0:
            continue

        # Modal Note Voting
        note_votes = {}

        for pitch, weight in zip(f0, voiced_prob):
            note = librosa.hz_to_note(pitch)
            note_votes[note] = note_votes.get(note, 0) + weight

        best_note = max(note_votes, key=note_votes.get)

        raw_segments.append({
            "start": start_sample,
            "end": end_sample,
            "note": best_note,
            "votes": note_votes
        })


    # Merge Fragmented Segments
    merged = []

    def note_to_midi(note):
        return librosa.note_to_midi(note)

    for seg in raw_segments:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]

        prev_midi = note_to_midi(prev["note"])
        curr_midi = note_to_midi(seg["note"])

        # Conditions to merge:
        same_note = prev["note"] == seg["note"] # note checking
        close_pitch = abs(prev_midi - curr_midi) <= 0  # 0 semitones apart (aka same note again)
        # small_gap = (seg["start"] - prev["end"]) < sr * 0.08  # ~80ms gap, needs work as this causes same note follow ups to merge
        small_gap = (len(seg) < sr * 0.08) or (len(prev) < sr * 0.08) # Alternative solution
        if (same_note or close_pitch) and small_gap: # Merge
            prev["end"] = seg["end"]

            # Modal Vote Merge
            for k, v in seg["votes"].items():
                prev["votes"][k] = prev["votes"].get(k, 0) + v

            prev["note"] = max(prev["votes"], key=prev["votes"].get)

        else:
            merged.append(seg)
    # --------------------------------------------------------
    final_notes = [
        (
            librosa.samples_to_time(seg["start"], sr=sr),
            librosa.samples_to_time(seg["end"], sr=sr),
            seg["note"]
        )
        for seg in merged
    ]

    return final_notes

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import numpy as np


def manual_piano_roll(note_events, bpm=None, window=6):
    fig, ax = plt.subplots(figsize=(12, 5))
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    # Notes into MIDI
    processed_notes = [
        (start, end, librosa.note_to_midi(note))
        for start, end, note in note_events
    ]

    # Pitch Range for MIDI 
    all_pitches = [p for _, _, p in processed_notes]

    if len(all_pitches) == 0:
        print("No notes to display.")
        return

    min_pitch = min(all_pitches)
    max_pitch = max(all_pitches)

    # Centre of notes
    center_pitch = (min_pitch + max_pitch) / 2

    # How many notes are visible
    pitch_window = 24   # 2 octaves 

    scroll_x = 0.0
    zoom = window

    def draw():
        ax.clear()

        left = scroll_x
        right = scroll_x + zoom

        y_min = center_pitch - pitch_window / 2
        y_max = center_pitch + pitch_window / 2

        for start, end, pitch in processed_notes:
            if end < left or start > right:
                continue

            rect = patches.Rectangle(
                (start, pitch - 0.4),
                end - start,
                0.8
            )
            ax.add_patch(rect)

        ax.set_xlim(left, right)
        ax.set_ylim(y_min, y_max)

        ax.set_title("MIDI Piano Roll")
        ax.set_xlabel("Time")
        ax.set_ylabel("Pitch")

        # Beat grid
        if bpm:
            beat_duration = 60 / bpm
            beats = np.arange(left, right, beat_duration)
            for b in beats:
                ax.axvline(b, linestyle='--', alpha=0.2)

        # Y ticks based on visible range
        yticks = np.arange(int(y_min), int(y_max), 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([librosa.midi_to_note(y) for y in yticks])

        ax.grid(True, axis='y', alpha=0.2)

        fig.canvas.draw_idle()

    # Scroll wheel handler
    def on_scroll(event):
        nonlocal scroll_x

        base_speed = 0.5

        if event.key == 'shift':
            speed = base_speed * 3
        else:
            speed = base_speed

        if event.button == 'up':
            scroll_x += speed
        elif event.button == 'down':
            scroll_x = max(0, scroll_x - speed)

        draw()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    draw()
    plt.show()
  


# MAIN -------------------------------------------------------------------------

# file_path = "data/metronome_test.wav"
# file_path = "data/3_note_piano_test.wav"
file_path = "data/midi_piano.wav"
# file_path = "data/voice_whistle.wav"
# file_path = "data/SOFT_PiANO.wav"
# file_path = "data/usertest1.wav"

try:
    y, sr = librosa.load(file_path, sr=None)
except Exception as e:
    print(f"failed to load audio file: {e}")
    y, sr = None, None

print("Audio Loaded Successfully")
print(f"Sample Rate: {sr}")
print(f"Audio Duration: {len(y) / sr:.2f} seconds")

# BPM Detection
bpm, onset_env = estimate_bpm(y, sr)
print(f"Estimated BPM: {bpm:.2f}")

# Beat Tracking
_, beat_frames = librosa.beat.beat_track(
    onset_envelope=onset_env,
    sr=sr,
    start_bpm=bpm
)

beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Note Detection
note_events = detect_note_events(y, sr)

manual_piano_roll(
    note_events,
    bpm=bpm,     # grid lines
    window=6     # visible time
)

"""
# Old Ouput Format (Not MIDI Piano Roll)
print("\nDetected Notes:")
for start, end, note in note_events:
    print(f"{start:.2f}s - {end:.2f}s → {note}")

# Plot Graph
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)

# Beats (red)
plt.vlines(beat_times, ymin=min(y), ymax=max(y), color='r', linestyle='--', label='Beats')

# Note Onsets (green)
note_starts = [start for start, _, _ in note_events]
plt.vlines(note_starts, ymin=min(y), ymax=max(y), color='g', linestyle='-', label='Notes')

plt.title(f"Waveform (BPM ≈ {bpm:.2f})")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
"""
