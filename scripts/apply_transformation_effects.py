# import numpy as np
# from pydub import AudioSegment
# from pydub.effects import low_pass_filter, high_pass_filter
# import math

# # Helper: scale function to normalize delta to effect range
# def scale(value, in_min, in_max, out_min, out_max):
    # value = max(min(value, in_max), in_min)  # clamp
    # return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

# # Main effect application

# def apply_transformation_effects(audio: AudioSegment,
                                 # original_features: dict,
                                 # transformed_features: dict) -> AudioSegment:
    # # Only include numeric keys
    # numeric_keys = ['rms', 'centroid', 'flatness', 'bandwidth', 'zcr', 'onset_density']
    # delta = {k: float(transformed_features[k]) - float(original_features[k]) for k in numeric_keys}

    # modified = audio

    # # --- 1. Volume adjustment from RMS ---
    # rms_delta = delta['rms']
    # gain_db = scale(rms_delta, -0.05, 0.05, -6, 6)  # max +/- 6 dB change
    # modified = modified.apply_gain(gain_db)

    # # --- 2. Brightness shift via EQ ---
    # centroid_delta = delta['centroid']
    # if centroid_delta > 0:
        # cutoff = scale(centroid_delta, 0, 3000, 2000, 4000)
        # modified = high_pass_filter(modified, cutoff)
    # else:
        # cutoff = scale(abs(centroid_delta), 0, 3000, 2000, 500)
        # modified = low_pass_filter(modified, cutoff)

    # # --- 3. Reverb effect from flatness (simple approximation with gain and EQ) ---
    # flatness_delta = delta['flatness']
    # if flatness_delta > 0.05:
        # modified = low_pass_filter(modified, 3000)
        # modified = modified.apply_gain(2)

    # return modified
import numpy as np
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import math

# Helper: scale function to normalize delta to effect range
def scale(value, in_min, in_max, out_min, out_max):
    value = max(min(value, in_max), in_min)  # clamp
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

# Main effect application
def apply_transformation_effects(audio: AudioSegment,
                                 original_features: dict,
                                 transformed_features: dict) -> AudioSegment:
    numeric_keys = ['rms', 'centroid', 'flatness', 'bandwidth', 'zcr', 'onset_density']
    delta = {k: float(transformed_features[k]) - float(original_features[k]) for k in numeric_keys}

    modified = audio

    # --- 1. Volume adjustment from RMS ---
    rms_delta = delta['rms']
    if abs(rms_delta) > 0.001:
        gain_db = scale(rms_delta, -0.05, 0.05, -6, 6)
        modified = modified.apply_gain(gain_db)

    # --- 2. Brightness shift via EQ from Spectral Centroid ---
    centroid_delta = delta['centroid']
    if abs(centroid_delta) > 50:
        if centroid_delta > 0:
            cutoff = scale(centroid_delta, 0, 3000, 2000, 4000)
            modified = high_pass_filter(modified, cutoff)
        else:
            cutoff = scale(abs(centroid_delta), 0, 3000, 2000, 500)
            modified = low_pass_filter(modified, cutoff)

    # --- 3. Reverb-like softening from Flatness ---
    flatness_delta = delta['flatness']
    if flatness_delta > 0.01:
        modified = low_pass_filter(modified, 3000)
        modified = modified.apply_gain(1.5)

    # --- 4. Subtle stereo widening from Bandwidth ---
    # Simulate by overlaying a slightly delayed copy (pseudo-width)
    bandwidth_delta = delta['bandwidth']
    if abs(bandwidth_delta) > 100:
        delay_ms = scale(bandwidth_delta, -1000, 1000, 5, 30)
        delayed = modified - 6  # quieter version
        delayed = delayed[0:len(modified) - int(delay_ms)]
        modified = modified.overlay(delayed, position=int(delay_ms))

    # --- 5. Rhythmic echo from Onset Density ---
    onset_delta = delta['onset_density']
    if onset_delta > 0.02:
        echo = modified - 9  # quieter echo
        echo = echo[100:]  # basic offset
        modified = modified.overlay(echo, loop=False)

    return modified

