import pyaudio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import librosa
import math
import scipy.io.wavfile as wavfile
import os

# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d

# Audio parameters
CHUNK = 4096  # Buffer size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample rate
UPDATE_INTERVAL = 20  # in millisec

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Set up the plot
# fig, ax = plt.subplots(figsize=(12, 4))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))


ax1.set_xlim(0, 4000)  # Focus on speech frequencies (0-4000 Hz)
ax1.set_ylim(0, 5)  # non dB scale
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Magnitude')
ax1.set_title('Real-time Voice FFT and Formant Display')
ax1.grid(True, alpha=0.3)


ax2.set_xlim(0, 4000)  # Focus on speech frequencies (0-4000 Hz)
ax2.set_ylim(0, 5)  # non dB scale
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude dB')

line1, = ax1.plot([], [], 'b-', linewidth=1, label='FFT')
line2, = ax2.plot([], [], 'b-', linewidth=1,alpha=0.7,) # label='dB Magnitude')
f0_line = ax2.axvline(x=0, color='blue', linestyle='--', linewidth=2)   #, label='F0 (Pitch)')
f1_line = ax2.axvline(x=0, color='red', linestyle='-', linewidth=2)     #, label='F1')
f2_line = ax2.axvline(x=0, color='green', linestyle='-', linewidth=2)   #, label='F2')
f3_line = ax2.axvline(x=0, color='orange', linestyle='-', linewidth=2)  #, label='F3')
f4_line = ax2.axvline(x=0, color='purple', linestyle='-', linewidth=2)  #, label='fF4')

ax2.legend(loc='upper right')

# # Text display for frequency values
freq_text = ax2.text(0.02,  # x
                     0.98,  # y
                     '',  # string
                     transform=ax1.transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round',
                               facecolor='white',
                               alpha=0.8))


def find_pitch(audio_data, sample_rate):
    p = []
    # 65 Hz = C2, 3520 Hz = A7
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, sr=sample_rate, fmin=65, fmax=3520)
    mask = voiced_flag == True
    pitch = np.average(f0[mask])
    return pitch


# def find_formants(spectrum, freqs, num_formants=4):
# audio_data  - input ndarray of windowed_data
# freqs       - FFT frequency buckets
# n_formants  - number of formants to find
# sample_rate - samples/sec - usually 44100 for WAV files
def find_formants(audio_data, freqs, n_formants=4, sample_rate=RATE):
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
    # print("lpc_coeffs = ", lpc_coeffs)

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


def update_plot(frame):
    """Update function for animation"""
    try:
        # Read audio data
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Convert to float and normalize
        audio_data = audio_data.astype(np.float32) / 32768.0

        # Apply window function
        windowed_data = audio_data * np.hanning(len(audio_data))

        # Compute FFT
        fft = np.fft.fft(windowed_data)
        magnitude = np.abs(fft)

        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        # Create frequency array
        freqs = np.fft.fftfreq(len(fft), 1 / RATE)

        # Take only positive frequencies
        positive_freqs = freqs[:len(freqs) // 2]
        positive_magnitude_db = magnitude_db[:len(magnitude_db) // 2]
        positive_magnitude = magnitude[:len(magnitude) // 2]

        # Update Plot 1: FFT Magnitude
        line1.set_data(positive_freqs, positive_magnitude)
        line2.set_data(positive_freqs, magnitude[:len(magnitude) // 2])

        # Plot 2: db Magnitude + Formants
        # Create a more detailed spectrum for formant visualization
        freqs_detailed = np.linspace(0, 4000, 1000)

        # Interpolate magnitude spectrum for smoother display
        magnitude_interp = np.interp(freqs_detailed, freqs, magnitude)
        magnitude_db = 20 * np.log10(magnitude_interp + 1e-10)

        # Find pitch (F0)
        f0 = find_pitch(audio_data, RATE)

        # Find formants
        formants, formant_bandwidth = find_formants(windowed_data, positive_freqs)
        # f1, f2, f3, f4 = formants
        print("formants=", formants[:5])
        f1 = formants[1]
        f2 = formants[2]
        f3 = formants[3]
        f4 = formants[4]

        # Update vertical lines
        f0_line.set_xdata([f0, f0])
        f1_line.set_xdata([f1, f1])
        f2_line.set_xdata([f2, f2])
        f3_line.set_xdata([f3, f3])
        f4_line.set_xdata([f4, f4])

        # ******************** UL PLOT 1 *******************
        # Update frequency text display - this appears in the Upper Left Corner of the First Plot
        #freq_info = f'F0: {f0:.0f} Hz\nF1: {f1:.0f} Hz\nF2: {f2:.0f} Hz\nF3: {f3:.0f} Hz\nF4: {f4:.0f} Hz'
        if not math.isnan(f0):
            freq_info = f'F0: {f0:>5.0f} Hz'
        else:
            freq_info = f'F0: {" ":>5} Hz'
        freq_info += f'\nF1: {f1:>5.0f} Hz\nF2: {f2:>5.0f} Hz\nF3: {f3:>5.0f} Hz\nF4: {f4:>5.0f} Hz'
        freq_text.set_text(freq_info)

        # Set line visibility based on whether frequency was detected
        f0_line.set_visible(f0 > 0)
        f1_line.set_visible(f1 > 0)
        f2_line.set_visible(f2 > 0)
        f3_line.set_visible(f3 > 0)
        f4_line.set_visible(f4 > 0)

        # UNCOMMENTING THESE LINES GIVES TOO MANY FORMANT LINES IN THE 2ND PLOT
        # Mark formants
        # formant_labels = ['F0', 'F1', 'F2', 'F3', 'F4']
        # colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        #
        # for i, (formant, bw, label, color) in enumerate(zip(formants, formant_bandwidth, formant_labels, colors)):
        #     #print("i=",i," formant=",formant," bw=",bw," label=",label," color=",color)
        #     if formant > 0:  # Only plot if formant was found
        #         if i == 0:  # Pitch
        #             if math.isnan(formant):
        #                 label = f'{label}:               Hz'
        #             else:
        #                 label = f'{label}: {formant:.0f} Hz'
        #             ax2.axvline(x=formant, color=color, linestyle='-', linewidth=2, label=label)
        #             print("i=",i," label=",label)
        #         else:
        #             ax2.axvline(x=formant, color=color, linestyle='-', linewidth=2,label=f'{label}: {formant:.0f} +/- {bw:.0f} Hz')
        #             #ax.axvline(x=formant, color=color, linestyle='-', linewidth=2,label=f'{label}: {formant:.0f} +/- {bw:.0f} Hz')
        #             print("i=", i," label=",f'{label}: {formant:.0f} +/- {bw:.0f} Hz')



    except Exception as e:
        print(f"Error in update_plot: {e}")

    return line1, line2, freq_text
    #return line1, line2, f0_line, f1_line, f2_line, f3_line, f4_line, freq_text
    # return line, f0_line, freq_text
    # return line, freq_text


def cleanup():
    """Cleanup function"""
    stream.stop_stream()
    stream.close()
    p.terminate()


# Create animation
################################## changed blit to False to force updating of formant lines #######
anim = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL, blit=False, cache_frame_data=False)



# Set up cleanup on window close
def on_close(event):
    cleanup()
    plt.close('all')


fig.canvas.mpl_connect('close_event', on_close)

print("Starting real-time voice spectrum analyzer...")
print("Speak into your microphone. The spectrum will update every 0.1 seconds.")
print("Close the window to stop the program.")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nProgram interrupted by user")
finally:
    cleanup()
