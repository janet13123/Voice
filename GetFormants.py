import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa
import os

# CONSTANTS USED
# the sample rate for the voice recording
RATE = 44100
# display formant bandwidths as dashed lines
DISP_BANDWIDTHS = False
# the recording time for the voice
UPDATE_INTERVAL = 0.5
# the number of formants to display
NUMBER_OF_FORMANTS = 4


# outfn - output filename
# duration - the time in seconds to record the voice
# rate - the sample rate to use
def record_voice(outfn, duration, rate):
    print(f"Recording for {duration} seconds...")
    print("Start speaking now!")
    filename = outfn
    sample_rate = rate

    # Record audio
    print("rate=", rate)
    print("sample_rate=", sample_rate)
    print(f"duration={duration}")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64', blocking=True)
    sd.wait()  # Wait until recording is finished

    recording_int16 = np.int16(recording * 32767).flatten()

    # Save to WAV file
    wavfile.write(filename, sample_rate, recording_int16)
    print(f"Audio saved to {filename}")

    return recording.flatten()

# filename - WAV file to load
# returns sample_rate, audio_data
def load_audio(filename):
    """Load audio from WAV file"""
    if os.path.exists(filename):
        sample_rate, audio_data = wavfile.read(filename)
        # Convert to float if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float64) / 32767.0
        return sample_rate, audio_data
    else:
        raise FileNotFoundError(f"WAV file {filename} not found")


# audio_data - ndarray of audio data read from WAV file
# sample_rate - samples/sec - usually 44100
# returns - freqs (fft bucket frequencies), magnitude (abs value of complex fft value)
def compute_fft(audio_data, sample_rate):
    """Compute FFT of audio data"""
    # Apply window function to reduce spectral leakage
    windowed_audio = audio_data * np.hanning(len(audio_data))

    # Compute FFT
    fft_result = np.fft.fft(windowed_audio)

    # Get magnitude spectrum (only positive frequencies)
    magnitude = np.abs(fft_result[:len(fft_result) // 2])

    # Create frequency axis
    freqs = np.fft.fftfreq(len(audio_data), 1 / sample_rate)[:len(fft_result) // 2]

    return freqs, magnitude


# audio_data - input ndarray
# n_formants - number of formants to find
# sample_rate - samples/sec - usually 44100 for WAV files
def find_formants(audio_data, n_formants, sample_rate):
    """
    Find formants using LPC analysis with librosa

    Args:
        audio_data: Audio signal
        n_formants: Number of formants to find

    Returns:
        formants: Array of formant frequencies
    """
    # Use librosa to compute LPC coefficients
    # Order is typically 2 + sample_rate/1000 for formant analysis
    lpc_order = int(2 + sample_rate / 1000)

    # Compute LPC coefficients using librosa
    lpc_coeffs = librosa.lpc(audio_data, order=lpc_order)
    #print("lpc_coeffs = ", lpc_coeffs)

    # Find roots of the LPC polynomial
    roots = np.roots(lpc_coeffs)

    # Convert roots to frequencies
    # Only consider roots inside the unit circle (stable poles)
    stable_roots = roots[np.abs(roots) < 1]

    # Bandwidth estimation
    bandwidths = -0.5 * (sample_rate / (2 * np.pi)) * np.log(np.abs(stable_roots))

    # Convert to frequencies
    angles = np.angle(stable_roots)
    freqs = angles * sample_rate / (2 * np.pi)

    # mask - Only keep positive frequencies
    mask = freqs > 0
    positive_freqs = freqs[mask]
    # use same mask for bandwidths
    pos_freq_bandwidths = bandwidths[mask]

    # get pitch as F0
    pitch = find_pitch(audio_data, sample_rate)

    #  need to zip sort these together - sort formants with it's bandwidth
    formants, formant_bandwidth = zip(*sorted(zip(positive_freqs, pos_freq_bandwidths)))
    # Sort frequencies
    # formants = np.sort(positive_freqs)

    # Return first n_formants formants
    if len(formants) >= n_formants:
        # insert pitch at the beginning of the list
        formants = np.insert(formants, 0, pitch)
        # insert 0 for bandwidth of pitch
        formant_bandwidth = np.insert(formant_bandwidth, 0, 0)
        return formants[:n_formants + 1], formant_bandwidth[:n_formants + 1]
    else:
        # Pad with zeros if not enough formants found
        padded_formants = np.zeros(n_formants + 1)
        padded_formant_bandwidth = np.zeros(n_formants + 1)
        padded_formant_bandwidth[:len(formant_bandwidth)] = formant_bandwidth
        padded_formant_bandwidth = np.insert(padded_formant_bandwidth, 0, 0)
        return padded_formants, padded_formant_bandwidth


# audio_data = the audio data values in an ndarray
# sample_rate - samples/sec - usually 44100 for WAV files
# returns the average pitch in the voiced parts of the sample
def find_pitch(audio_data, sample_rate):
    p = []
    # p = librosa.yin(audio_data, fmin=80, fmax=4000, sr=sample_rate)
    # 65 Hz = C2, 3520 Hz = A7
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, sr=sample_rate, fmin=65, fmax=3520)
    mask = voiced_flag == True
    pitch = np.mean(f0[mask])
    return pitch

# audio_data - the audio data values in an ndarray
# num_formants - number of formants to try and display
# sample_rate - samples/sec - usually 44100 for WAV files
# filename - The name of the WAV file where the data came from
def plot_analysis(audio_data, num_formants, sample_rate, filename):
    """Plot FFT and formants analysis"""
    # Compute FFT
    freqs, magnitude = compute_fft(audio_data, sample_rate)

    # Find formants, bandwidths
    formants, formant_bandwidth = find_formants(audio_data, num_formants, sample_rate)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Plot 1: FFT Magnitude Spectrum
    ax1.plot(freqs, magnitude)  # Convert to dB
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('FFT Magnitude Spectrum File:' + filename)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4000)  # Focus on speech frequency range

    # Plot 2: Formants
    # Create a more detailed spectrum for formant visualization
    freqs_detailed = np.linspace(0, 4000, 1000)

    # Interpolate magnitude spectrum for smoother display
    magnitude_interp = np.interp(freqs_detailed, freqs, magnitude)
    magnitude_db = 20 * np.log10(magnitude_interp + 1e-10)

    ax2.plot(freqs_detailed, magnitude_db, 'b-', alpha=0.7, label='Spectrum')

    # Mark formants
    formant_labels = ['F0', 'F1', 'F2', 'F3', 'F4']
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    if DISP_BANDWIDTHS:
        for i, (formant, bw, label, color) in enumerate(zip(formants, formant_bandwidth, formant_labels, colors)):
            if formant > 0:  # Only plot if formant was found
                if i == 0:  # Pitch
                    ax2.axvline(x=formant, color=color, linestyle='-', linewidth=2, label=f'{label}: {formant:.0f} Hz')
                else:
                    ax2.axvline(x=formant, color=color, linestyle='-', linewidth=2,
                                label=f'{label}: {formant:.0f} +/- {bw:.0f} Hz')
                    ax2.axvline(x=formant + bw, color=color, linestyle='--', linewidth=2, alpha=0.4)
                    ax2.axvline(x=formant - bw, color=color, linestyle='--', linewidth=2, alpha=0.4)
    else:
        for i, (formant, bw, label, color) in enumerate(zip(formants, formant_bandwidth, formant_labels, colors)):
            if formant > 0:  # Only plot if formant was found
                if i == 0:  # Pitch
                    ax2.axvline(x=formant, color=color, linestyle='-', linewidth=2, label=f'{label}: {formant:.0f} Hz')
                else:
                    ax2.axvline(x=formant, color=color, linestyle='-', linewidth=2,
                                label=f'{label}: {formant:.0f} +/- {bw:.0f} Hz')

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Formant Analysis (F0, F1, F2, F3, F4)')
    ax2.grid(True, alpha=0.3)

    ax2.legend()
    plt.legend(loc='upper right')

    ax2.set_xlim(0, 4000)

    plt.tight_layout()
    plt.show()

    # Print formant values
    print("Filename: " + filename)
    print("\nFormant Analysis Results:")
    for i, formant in enumerate(formants):
        if formant > 0:
            formant_bw = formant_bandwidth[i]
            print(f"F{i}: {formant:.2f} +/- {formant_bw:.2f} Hz")
        else:
            print(f"F{i}: Not detected")
    return


if __name__ == "__main__":

    print("Voice Recording and Formant Analysis")
    print("=" * 40)
    sample_rate = RATE
    duration = UPDATE_INTERVAL
    # outfn = "G3M2OQHL1s.wav"
    # outfn = "G3M1OQHL.wav"
    outfn = "TEMP.wav"
    num_formants = NUMBER_OF_FORMANTS

    #yn = input("Record a voice and show analysis? :")[0].upper()
    yn = "Y"

    if yn != 'Y':
        print("Loading WAV file...")

        # Load audio from file instead of recording voice
        # outfn = "G3M1OQHL.wav"
        # outfn = "G3M2OQHL1s.wav"
        # outfn = "TEMP.WAV"
        _, audio_data = load_audio(outfn)
        print("len(audio_data)=",len(audio_data))

        # Perform analysis and plotting
        print("starting plot")
        plot_analysis(audio_data, num_formants, sample_rate, outfn)
        print("finished plot")

        pitch = find_pitch(audio_data, sample_rate)
        print(f"Pitch = {pitch:.2f}")
    else:
        # Record voice
        print("Recording Voice...")
        try:
            recorded_audio = record_voice(outfn, duration, sample_rate)

            # Load audio from file (to verify save/load works)
            _, audio_data = load_audio(outfn)

            # Perform analysis and plotting
            plot_analysis(audio_data, num_formants, sample_rate, outfn)

            pitch = find_pitch(audio_data, sample_rate)
            print(f"Pitch = {pitch:.2f}")


        except Exception as e:
            print(f"Error occurred: {e}")
            print("Make sure you have a microphone connected and try again.")

