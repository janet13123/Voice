import tkinter as tk
from tkinter import ttk
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm
import librosa
import queue


class VoiceViewGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio FFT and Formant Viewer")
        # handles window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Audio parameters for PyAudio
        self.format = pyaudio.paFloat32  # 32 bit float
        self.sample_rate = 44100
        self.channels = 1
        # initial button settings
        self.chunk_size = 4096
        self.update_rate = 0.1  # in seconds
        self.freq_max = 5000

        self.num_formants = 4

        # PyAudio objects
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False

        # Data storage
        # fill with zeros
        self.audio_data = np.zeros(self.chunk_size)
        # a queue to store the audio data
        self.data_queue = queue.Queue()

        # Setup the buttons and canvas for the plot
        self.setup_gui()
        # contains the FuncAnimation and line setup stuff
        self.setup_plots()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH,           # fill if widget grows
                        expand=True,            # fill if parent size grows
                        padx=10, pady=10)

        # Left side - Controls
        left_frame = ttk.Frame(main_frame)      # main_frame contains left_frame
        left_frame.pack(side=tk.LEFT,           # add to LEFT side
                        fill=tk.Y,
                        padx=(0, 10))


        # # ADD COMBO BOXES WITH DROPDOWNS
        # # Chunk Size control - disabled for now
        # ttk.Label(left_frame,
        #           text="Chunk Size:").pack(pady=5)
        # # with StringVar, we can assign a 'master','value',and 'name' to the string
        # self.chunk_var = tk.StringVar(value=str(self.chunk_size))
        # # Combobox is a text field with a drop down list of values to select
        # chunk_combo = ttk.Combobox(
        #     left_frame,                         # add to left_frame
        #     textvariable=self.chunk_var,                # fill the textbox with this value
        #     values=['1024', '2048', '4096', '8192'],    # drop down list values
        #     state='readonly',
        #     width=10)
        # chunk_combo.pack(pady=5)
        # chunk_combo.bind('<<ComboboxSelected>>',        # virtual event
        #                  self.update_chunk_size)        # call this function when selected

        # Update Rate control - disabld for now
        # ttk.Label(left_frame,
        #           text="Update Rate (s):").pack(pady=5)
        # self.rate_var = tk.StringVar(value=str(self.update_rate))
        # print("rate_var = ",self.rate_var.get())
        # rate_combo = ttk.Combobox(left_frame,
        #                           textvariable=self.rate_var,
        #                           values=['0.1', '0.2', '0.5', '1.0'],
        #                           state='readonly',
        #                           width=10)
        # rate_combo.pack(pady=5)
        # rate_combo.bind('<<ComboboxSelected>>',
        #                 self.update_rate_change)

        # freq limit control
        ttk.Label(left_frame, text="Freq Limit (Hz):").pack(pady=5)
        self.freq_max_var = tk.StringVar(value=str(self.freq_max))
        freq_combo = ttk.Combobox(left_frame,
                                  textvariable=self.freq_max_var,
                                  values=['500', '1000', '3000', '5000', '10000', '20000'],
                                  state='readonly',
                                  width=10)
        freq_combo.pack(pady=5)
        freq_combo.bind('<<ComboboxSelected>>',
                        self.update_freq_max)


        # BUTTONS
        # Start button - start_recording
        self.start_btn = ttk.Button(left_frame,
                                    text="Start",
                                    command=self.start_recording,
                                    width=12)
        self.start_btn.pack(pady=10)

        # Pause button - pause_recording
        self.pause_btn = ttk.Button(left_frame,
                                    text="Pause",
                                    command=self.pause_recording,
                                    state='disabled',
                                    width=12)
        self.pause_btn.pack(pady=5)


        # PLOTS
        # Right side - Plots
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)

        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_plots(self):
        # Initialize frequency array
        self.freqs = np.fft.fftfreq(self.chunk_size, 1 / self.sample_rate)
        self.freqs = self.freqs[:self.chunk_size // 2]

        # Setup first subplot (magnitude)
        self.ax1.set_title("FFT Magnitude")
        self.ax1.set_xlabel("Frequency (Hz)")
        self.ax1.set_ylabel("Magnitude")
        # self.ax1.set_xlim(0, self.sample_rate / 2)
        self.ax1.set_xlim(0, self.freq_max)
        self.ax1.set_ylim(0, 10)
        self.ax1.grid(True, alpha=0.3)
        # Why are we passing an array of 0's ?
        self.line1, = self.ax1.plot(self.freqs, np.zeros(len(self.freqs)), 'b-')

        # Setup second subplot (decibels)
        self.ax2.set_title("FFT Magnitude (dB)")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Magnitude (dB)")
        # self.ax2.set_xlim(0, self.sample_rate / 2)
        self.ax2.set_xlim(0, self.freq_max)
        self.ax2.set_ylim(-80, 20)
        self.ax2.grid(True, alpha=0.3)
        # Here we are passing an array of -80's
        self.line2, = self.ax2.plot(self.freqs, np.full(len(self.freqs), -80), 'r-')
        self.vline0 = self.ax2.axvline(0, color='blue', linestyle='--', linewidth=2, label='F0:')
        self.vline1 = self.ax2.axvline(0, color='red', linestyle='-', linewidth=2, label='F1:')
        self.vline2 = self.ax2.axvline(0, color='green', linestyle='-', linewidth=2, label='F2:')
        self.vline3 = self.ax2.axvline(0, color='orange', linestyle='-', linewidth=2, label='F3:')
        self.vline4 = self.ax2.axvline(0, color='purple', linestyle='-', linewidth=2, label='F4:')
        font_props = fm.FontProperties(family='monospace', size=10, weight='bold')
        self.my_legend = self.ax2.legend(loc='upper right', prop=font_props)

        # Initialize animation
        self.animation = FuncAnimation(self.fig,                           # the figure to update
                                       self.update_plots,                 # the update function
                                       interval=int(self.update_rate * 1000),   # update interval in ms
                                       blit=False,                              # optimize redrawing ?
                                       cache_frame_data=False)                  # is data cached ?
        self.animation.pause()

    # # disable this button for now
    # def update_chunk_size(self, event=None):
    #     if not self.is_recording:
    #         self.chunk_size = int(self.chunk_var.get())
    #         # fill audio_data with 0's
    #         self.audio_data = np.zeros(self.chunk_size)
    #         self.freqs = np.fft.fftfreq(self.chunk_size, 1 / self.sample_rate)
    #         self.freqs = self.freqs[:self.chunk_size // 2]
    #
    #         # Update plot lines
    #         self.line1.set_xdata(self.freqs)
    #         self.line1.set_ydata(np.zeros(len(self.freqs)))
    #         self.line2.set_xdata(self.freqs)
    #         self.line2.set_ydata(np.full(len(self.freqs), -80))
    #
    #         # recompute data limits for plots
    #         self.ax1.relim()
    #         # autoscale view limits
    #         self.ax1.autoscale_view()
    #         self.canvas.draw()

    # # cant change rate - animation update rate is ignored
    # def update_rate_change(self, event=None):
    #     # self.update_rate = float(self.rate_var.get())
    #     # print("update_rate = ", self.update_rate)
    #     # # update animation rate
    #     # print("has attr = ",hasattr(self, 'animation'))
    #     # if hasattr(self, 'animation'):
    #     #     self.animation.event_source.interval = int(self.update_rate * 1000)
    #     # print("ANIMATION update_rate = ", self.update_rate)
    #     #TRY THIS - didn't seem to work .......
    #     if not self.is_recording:
    #         self.animation.event_source.stop()
    #         self.update_rate = float(self.rate_var.get())
    #         self.audio_data = np.zeros(self.chunk_size)
    #         self.freqs = np.fft.fftfreq(self.chunk_size, 1 / self.sample_rate)
    #         self.freqs = self.freqs[:self.chunk_size // 2]
    #         self.line1.set_xdata(self.freqs)
    #         self.line1.set_ydata(np.zeros(len(self.freqs)))
    #         self.line2.set_xdata(self.freqs)
    #         self.line2.set_ydata(np.full(len(self.freqs), -80))
    #         self.ax1.relim()
    #         self.ax1.autoscale_view()
    #         self.canvas.draw()
    #         self.animation.resume()

    def update_freq_max(self, event=None):
        if not self.is_recording:
            self.freq_max = int(self.freq_max_var.get())
            self.audio_data = np.zeros(self.chunk_size)
            self.freqs = np.fft.fftfreq(self.chunk_size, 1 / self.sample_rate)
            self.freqs = self.freqs[:self.chunk_size // 2]

            # Update plot lines
            self.ax1.set_xlim(0, self.freq_max)
            self.ax2.set_xlim(0, self.freq_max)
            self.line1.set_xdata(self.freqs)
            self.line1.set_ydata(np.zeros(len(self.freqs)))
            self.line2.set_xdata(self.freqs)
            self.line2.set_ydata(np.full(len(self.freqs), -80))

            self.ax1.relim()
            self.ax1.autoscale_view()
            self.canvas.draw()

    def audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_data = np.frombuffer(in_data, dtype=np.float32)
        try:
            self.data_queue.put_nowait(self.audio_data.copy())
        except queue.Full:
            pass
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        try:
            self.stream = self.p.open(
                format=self.format,         # paFloat32
                channels=self.channels,     # 1 channel
                rate=self.sample_rate,      # 44100
                input=True,                 # this is an input stream
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback  # for non-blocking operation
            )

            self.is_recording = True
            self.stream.start_stream()
            self.animation.resume()

            self.start_btn.config(state='disabled')
            self.pause_btn.config(state='normal')
            #self.freq_combo.config(state = tk.NORMAL)

        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not start audio stream: {str(e)}")

    def pause_recording(self):
        if self.stream:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        self.animation.pause()

        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')

        # Clear the queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break


    # return True if voice is present
    # audio_data = np array of audio data
    # threshold_zrc = zero crossing rate threshold
    # threshold_energy = rms energy
    def detect_voice(self,audio_data, threshold_zcr=0.1, threshold_energy=0.01):

        # THIS ACCEPTS A WAV FILE INPUT, NOT A BUFFER #
        #y, sr = librosa.load(self.audio_data, sr=None)
        y = self.audio_data
        sr = self.sample_rate

        # Calculate Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # Calculate Root Mean Square (RMS) energy
        rms = librosa.feature.rms(y=y)[0]

        # Simple voice detection based on thresholds
        voice_frames_zcr = zcr < threshold_zcr
        voice_frames_energy = rms > threshold_energy

        # both conditions must be met to return True
        return np.any(voice_frames_zcr & voice_frames_energy)


    # def find_formants(spectrum, freqs, num_formants=4):
    # audio_data  - input ndarray of windowed_data
    # returns pitch, F1,F2,F3,F4   and  bandwidths
    def find_formants(self, audio_data):

        n_formants = self.num_formants

        if self.detect_voice(self.audio_data):
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
            lpc_order = int(2 + self.sample_rate / 1000)

            # Compute LPC coefficients using librosa
            lpc_coeffs = librosa.lpc(self.audio_data, order=lpc_order)

            #if lpc_coeffs contain INF or NaN it will cause an error when np.roots is called
            if np.any(np.isnan(lpc_coeffs)):
                #print(lpc_coeffs)
                padded_formants = np.zeros(n_formants + 1)
                padded_formant_bandwidth = np.zeros(n_formants + 1)
                return padded_formants, padded_formant_bandwidth
            if np.any(np.isinf(lpc_coeffs)):
                #print(lpc_coeffs)
                padded_formants = np.zeros(n_formants + 1)
                padded_formant_bandwidth = np.zeros(n_formants + 1)
                return padded_formants, padded_formant_bandwidth

            # Find roots of the LPC polynomial
            roots = np.roots(lpc_coeffs)

            # Convert roots to frequencies
            # Only consider roots inside the unit circle (stable poles)
            stable_roots = roots[np.abs(roots) < 1]
            # testing - see if unstable poles EVER occur
            unstable_roots = roots[np.abs(roots) >= 1]
            if len(unstable_roots):
                #print("UNSTABLE ROOTS = ", unstable_roots)
                padded_formants = np.zeros(n_formants + 1)
                padded_formant_bandwidth = np.zeros(n_formants + 1)
                return padded_formants, padded_formant_bandwidth


            # Bandwidth estimation
            #if np.abs(stable_roots).all() == 0:
            if np.all(np.abs(stable_roots) == 0):
                # ALL ZEROS - can't find bandwidths
                padded_formants = np.zeros(n_formants + 1)
                padded_formant_bandwidth = np.zeros(n_formants + 1)
                return padded_formants, padded_formant_bandwidth
            else:
                bandwidths = -0.5 * (self.sample_rate / (2 * np.pi)) * np.log(np.abs(stable_roots))

            # Convert to frequencies
            angles = np.angle(stable_roots)
            freqs = angles * self.sample_rate / (2 * np.pi)

            # mask - Only keep positive frequencies
            mask = freqs > 0
            positive_freqs = freqs[mask]

            # use same mask for bandwidths
            pos_freq_bandwidths = bandwidths[mask]

            # get pitch as F0, if no pitch, then 0
            pitch = self.find_pitch(self.audio_data)

            #  need to zip sort these together - sort formants with it's bandwidth
            formants, formant_bandwidth = zip(*sorted(zip(positive_freqs, pos_freq_bandwidths)))

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
        else:
            no_data = np.zeros(5)
            return no_data,no_data


    def find_pitch(self, audio_data):
        # returns average pitch, or 0 if cant find pitch
        # 88 Hz = G2, 330 Hz = E4
        fmin = 88
        fmax = 330

        f0, voiced_flag, _ = librosa.pyin(self.audio_data, sr=self.sample_rate,
                                          fmin=fmin, fmax=fmax,
                                          frame_length=self.chunk_size, fill_na=0)
        if len(f0[voiced_flag]) > 0:
            return np.average(f0[voiced_flag])
        else:
            return 0

    def update_plots(self, frame):
        if not self.is_recording:
            return self.line1, self.line2, self.vline0, self.vline1, self.vline2, self.vline3, self.vline4

        # Get latest audio data
        try:
            while not self.data_queue.empty():
                self.audio_data = self.data_queue.get_nowait()
        except self.queue.Empty:
            pass

        # Compute FFT
        if len(self.audio_data) == self.chunk_size:
            # Apply window function
            windowed = self.audio_data * np.hanning(self.chunk_size)

            # Compute FFT
            fft = np.fft.fft(windowed)
            fft_magnitude = np.abs(fft[:self.chunk_size // 2])

            # Update magnitude plot
            self.line1.set_ydata(fft_magnitude)
            self.ax1.relim()
            self.ax1.autoscale_view()

            # Convert to decibels and update dB plot
            fft_db = 20 * np.log10(np.maximum(fft_magnitude, 1e-10))
            self.line2.set_ydata(fft_db)

            formants, formant_bandwidth = self.find_formants(windowed)
            # f0, f1, f2, f3, f4 = formants, f0 = pitch, f1-f4 = formants
            # f0 = int(formants[0]) if not math.isnan(formants[0]) else None
            f0 = int(formants[0]) if not 0 else None
            f1 = int(formants[1])
            f2 = int(formants[2])
            f3 = int(formants[3])
            f4 = int(formants[4])

            bw0 = int(formant_bandwidth[0])
            bw1 = int(formant_bandwidth[1])
            bw2 = int(formant_bandwidth[2])
            bw3 = int(formant_bandwidth[3])
            bw4 = int(formant_bandwidth[4])

            self.vline0.set_xdata([f0, f0])
            self.vline1.set_xdata([f1, f1])
            self.vline2.set_xdata([f2, f2])
            self.vline3.set_xdata([f3, f3])
            self.vline4.set_xdata([f4, f4])

            s = ' '
            #BW = BandWidth of formant
            p = 'BW:'

            s0a = f'F0:{s:14}'
            s0b = f'F0:{f0:>5d}{s:>5}{s:>4}'
            s1 = f'F1:{f1:>5d}{p:^5}{bw1:>4d}'
            s2 = f'F2:{f2:>5d}{p:^5}{bw2:>4d}'
            s3 = f'F3:{f3:>5d}{p:^5}{bw3:>4d}'
            s4 = f'F4:{f4:>5d}{p:^5}{bw4:>4d}'

            if f0 == 0:
                self.my_legend.get_texts()[0].set_text(s0a)
            else:
                self.my_legend.get_texts()[0].set_text(s0b)
            self.my_legend.get_texts()[1].set_text(s1)
            self.my_legend.get_texts()[2].set_text(s2)
            self.my_legend.get_texts()[3].set_text(s3)
            self.my_legend.get_texts()[4].set_text(s4)

            # ******************** UL PLOT 1 *******************
            # Update frequency text display - this appears in the Upper Left Corner of the First Plot
            # freq_info = f'F0: {f0:.0f} Hz\nF1: {f1:.0f} Hz\nF2: {f2:.0f} Hz\nF3: {f3:.0f} Hz\nF4: {f4:.0f} Hz'
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

        return self.line1, self.line2, self.vline0, self.vline1, self.vline2, self.vline3, self.vline4

    def on_closing(self):
        self.pause_recording()
        if hasattr(self, 'p'):
            self.p.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VoiceViewGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
