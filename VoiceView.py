import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.io.wavfile as wavfile
import pyaudio
import librosa
import time
import os
import math
# TODO - ADD COLORS TO FREQ_TEXT
# TODO - LOOK AT SPEEDING UP THE CODE - UPDATE TIME ~ 0.2 SEC

# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d

# Audio parameters
RATE = 44100                # Sample rate
UPDATE_INTERVAL = 20        # in millisec
CHUNK = 8192                # Buffer size
FORMAT = pyaudio.paInt16
CHANNELS = 1


# Initialize PyAudio
p = pyaudio.PyAudio()

# Open microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))


ax1.set_xlim(0, 4000)  # Focus on speech frequencies (0-4000 Hz)
ax1.set_ylim(0, 10)
#ax1.set_xlabel('Frequency (Hz)') - not needed
ax1.set_ylabel('Magnitude')
ax1.set_title('Real-time Voice FFT')
ax1.grid(True, alpha=0.3)

ax2.set_xlim(0, 4000)  # Focus on speech frequencies (0-4000 Hz)
ax2.set_ylim(-80, 20)  # non dB scale
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude dB')
ax2.set_title('Real-time Voice Formants')
ax2.grid(True, alpha=0.3)

line1, = ax1.plot([], [], 'b-', linewidth=1, label='FFT')
line2, = ax2.plot([], [], 'b-', linewidth=1,alpha=0.7,)                 # label='dB Magnitude')
vline0 = ax2.axvline(0, color='blue', linestyle='--', linewidth=2,label='F0:')
vline1 = ax2.axvline(0, color='red', linestyle='-', linewidth=2, label='F1:')
vline2 = ax2.axvline(0, color='green', linestyle='-', linewidth=2, label='F2:')
vline3 = ax2.axvline(0, color='orange', linestyle='-', linewidth=2, label='F3:')
vline4 = ax2.axvline(0, color='purple', linestyle='-', linewidth=2, label='F4:')
# f0_line = ax2.axvline(x=0, color='blue', linestyle='--', linewidth=2)   #, label='F0 (Pitch)')
# f1_line = ax2.axvline(x=0, color='red', linestyle='-', linewidth=2)     #, label='F1')
# f2_line = ax2.axvline(x=0, color='green', linestyle='-', linewidth=2)   #, label='F2')
# f3_line = ax2.axvline(x=0, color='orange', linestyle='-', linewidth=2)  #, label='F3')
# f4_line = ax2.axvline(x=0, color='purple', linestyle='-', linewidth=2)  #, label='fF4')

my_legend = ax2.legend(loc='upper right')

# # Text display for frequency values
# freq_text = ax2.text(+1.01,  # x
#                      +0.98,  # y
#                      '',  # string
#                      transform=ax2.transAxes,
#                      fontsize=10,
#                      verticalalignment='top',
#                      bbox=dict(boxstyle='round',
#                                facecolor='white',
#                                alpha=0.8))

def find_pitch(audio_data, sample_rate):
    p = []
    # 65 Hz = C2, 3520 Hz = A7
    f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, sr=sample_rate, fmin=65, fmax=3520)
    mask = voiced_flag == True
    if np.sum(f0[mask])>0:
        pitch = np.average(f0[mask])
        return pitch
    else:
        return None


# def find_formants(spectrum, freqs, num_formants=4):
# audio_data  - input ndarray of windowed_data
# freqs       - FFT frequency buckets
# n_formants  - number of formants to find
# sample_rate - samples/sec - usually 44100 for WAV files
# returns pitch, F1,F2,F3,F4
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

    # Find roots of the LPC polynomial
    roots = np.roots(lpc_coeffs)

    # Convert roots to frequencies
    # Only consider roots inside the unit circle (stable poles)
    stable_roots = roots[np.abs(roots) < 1]
    # testing - see if unstable poles EVER occur
    unstable_roots = roots[np.abs(roots) >= 1]
    # if len(unstable_roots):
    #     print("UNSTABLE ROOTS = ",unstable_roots)

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


# frame - frame number from FuncAnimate, just a sequential integer, not used
def update_plot(frame):
    """Update function for animation"""
    start_update = time.time()
    try:
        # Read audio data
        start_read = time.time()
        data = stream.read(CHUNK, exception_on_overflow=False)
        end_read = time.time()
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
        line2.set_data(positive_freqs, magnitude_db[:len(magnitude_db) // 2])

        # Plot 2: db Magnitude + Formants
        # Create a more detailed spectrum for formant visualization
        freqs_detailed = np.linspace(0, 4000, 1000)

        # Interpolate magnitude spectrum for smoother display
        mask = freqs>0
        magnitude_interp = np.interp(freqs_detailed, freqs[mask], magnitude[mask])
        magnitude_db = 20 * np.log10(magnitude_interp + 1e-10)

        # Find formants
        formants = []
        formants, formant_bandwidth = find_formants(windowed_data, positive_freqs)

        # f0, f1, f2, f3, f4 = formants, f0 = pitch, f1-f4 = formants
        f0 = int(formants[0]) if not math.isnan(formants[0]) else None
        f1 = int(formants[1])
        f2 = int(formants[2])
        f3 = int(formants[3])
        f4 = int(formants[4])

        bw0 = int(formant_bandwidth[0])
        bw1 = int(formant_bandwidth[1])
        bw2 = int(formant_bandwidth[2])
        bw3 = int(formant_bandwidth[3])
        bw4 = int(formant_bandwidth[4])

        # Update vertical lines
        # f0_line.set_xdata([f0, f0])
        # f1_line.set_xdata([f1, f1])
        # f2_line.set_xdata([f2, f2])
        # f3_line.set_xdata([f3, f3])
        # f4_line.set_xdata([f4, f4])

        #f0_vline = f1_vline = f2_vline = f3_vline = f4_vline = float(0)
        # vline0 = ax.axvline(0, color='blue', linestyle='--', linewidth=2, label='F0 (Pitch)')
        vline0.set_xdata([f0, f0])
        #f0_vline = vline0.set_xdata([f0, f0])
        # vline1 = ax.axvline(0, color='red', linestyle='-', linewidth=2, label='F1')
        vline1.set_xdata([f1, f1])
        #f1_vline = vline1.set_xdata([f1, f1])
        # vline2 = ax.axvline(0, color='green', linestyle='-', linewidth=2, label='F2')
        vline2.set_xdata([f2, f2])
        #f2_vline = vline2.set_xdata([f2, f2])
        # vline3 = ax.axvline(0, color='orange', linestyle='-', linewidth=2, label='F3')
        #f3_vline = vline3.set_xdata([f3, f3])
        vline3.set_xdata([f3, f3])
        # vline4 = ax.axvline(0, color='purple', linestyle='-', linewidth=2, label='F4')
        #f4_vline = vline4.set_xdata([f4, f4])
        vline4.set_xdata([f4, f4])

        if f0 == None:
            my_legend.get_texts()[0].set_text(f'F0:                ')
        else:
            #my_legend.get_texts()[0].set_text(f'F0: {f0:.0f} +/- {bw0:.0f}')
            my_legend.get_texts()[0].set_text(f'F0:{f0:>5d} +/-{bw0:>5d}')
        #my_legend.get_texts()[1].set_text(f'F1: {f1:.0f} +/- {bw1:.0f}')
        my_legend.get_texts()[1].set_text(f'F1:{f1:>5d} +/-{bw1:>5d}')
        #my_legend.get_texts()[2].set_text(f'F2: {f2:.0f} +/- {bw2:.0f}')
        my_legend.get_texts()[2].set_text(f'F2:{f2:>5d} +/-{bw2:>5d}')
        #my_legend.get_texts()[3].set_text(f'F3: {f3:.0f} +/- {bw3:.0f}')
        my_legend.get_texts()[3].set_text(f'F3:{f3:>5d} +/-{bw3:>5d}')
        #my_legend.get_texts()[4].set_text(f'F4: {f4:.0f} +/- {bw4:.0f}')
        my_legend.get_texts()[4].set_text(f'F4:{f4:>5d} +/-{bw4:>5d}')

        # ******************** UL PLOT 1 *******************
        # Update frequency text display - this appears in the Upper Left Corner of the First Plot
        #freq_info = f'F0: {f0:.0f} Hz\nF1: {f1:.0f} Hz\nF2: {f2:.0f} Hz\nF3: {f3:.0f} Hz\nF4: {f4:.0f} Hz'
        # if not math.isnan(f0):
        #     freq_info = f'F0: {f0:>5.0f} Hz'
        # else:
        #     freq_info = f'F0: {" ":>5} Hz'
        # freq_info += f'\nF1: {f1:>5.0f} Hz\nF2: {f2:>5.0f} Hz\nF3: {f3:>5.0f} Hz\nF4: {f4:>5.0f} Hz'
        # freq_text.set_text(freq_info)

        # Set line visibility based on whether frequency was detected
        # f0_line.set_visible(f0 > 0)
        # f1_line.set_visible(f1 > 0)
        # f2_line.set_visible(f2 > 0)
        # f3_line.set_visible(f3 > 0)
        # f4_line.set_visible(f4 > 0)

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
        #
        #             ax2.axvline(x=formant, color=color, linestyle='-', linewidth=2, label=label)
        #         else:
        #             ax2.axvline(x=formant, color=color, linestyle='-', linewidth=2,label=f'{label}: {formant:.0f} +/- {bw:.0f} Hz')

    except Exception as e:
        print(f"Error in update_plot: {e}")

    finish_update = time.time()
    return line1, line2, vline0, vline1, vline2, vline3, vline4, my_legend

def cleanup():
    """Cleanup function"""
    stream.stop_stream()
    stream.close()
    p.terminate()

# Create animation
# Set blit to False to force updating of formant lines - maybe there's a better way to erase them?
anim = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL, blit=True, cache_frame_data=False)

# Set up cleanup on window close
def on_close(event):
    cleanup()
    print("RATE = ",RATE)
    print("CHUNK = ",CHUNK)
    print("UPDATE_INTERVAL = ", UPDATE_INTERVAL," ms")
    print("Read Time = ",CHUNK/RATE," ms")
    print("Suggested CHUNK size = ",RATE * UPDATE_INTERVAL / 1000)
    plt.close('all')

fig.canvas.mpl_connect('close_event', on_close)

print("Starting real-time voice spectrum analyzer...")
print(f"Speak into your microphone. The spectrum will update every {UPDATE_INTERVAL/1000} seconds.")
print("Close the window to stop the program.")

try:
    plt.show()
except KeyboardInterrupt:
    print("\nProgram interrupted by user")
finally:
    cleanup()
