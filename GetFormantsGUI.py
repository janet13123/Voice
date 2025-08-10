import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import GetFormants

# TODO - test on linux and mac also
# TODO - test that we don't overwrite non-WAV files
# TODO - does DISP_BANDWIDTHS == True even work ???

# CONSTANTS
RATE = 44100  # the default sample rate for the voice recording
DISP_BANDWIDTHS = False  # display formant bandwidths as dashed lines
DISP_BANDWIDTHS = True  # display formant bandwidths as dashed lines
REC_TIME = 0.5  # the recording time for the voice
NUMBER_OF_FORMANTS = 4  # the number of formants to display
MAX_FREQ = 4000  # the max freq to plot
DISP_BANDWIDTHS = False  # if True then also display the formant bandwidths in plot #3


class GetFormantsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Processing GUI")
        self.root.geometry("1200x700")

        # Variables
        self.mode_var = tk.StringVar(value="Record Voice")
        self.selected_file = tk.StringVar()
        self.filename_var = tk.StringVar()
        self.sample_rate = RATE
        self.n_samples = None
        self.time_length = None
        self.num_formants = NUMBER_OF_FORMANTS
        # default filename to record voice
        self.filename_var.set("TEMP.WAV")
        self.current_directory = tk.StringVar(value=os.getcwd())

        # Create main frames
        self.create_main_frames()

        # Create left panel components
        self.create_left_panel()

        # Create right panel with matplotlib
        self.create_right_panel()

        # Populate file list initially
        self.update_file_list()

    def create_main_frames(self):
        """Create the main left and right frames"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for controls
        self.left_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Right frame for plots
        self.right_frame = ttk.LabelFrame(main_frame, text="Plots", padding="10")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def create_left_panel(self):
        """Create all components for the left panel"""
        # Radio buttons for mode selection
        mode_frame = ttk.LabelFrame(self.left_frame, text="Mode Selection", padding="5")
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(mode_frame,
                        text="Record Voice",
                        variable=self.mode_var,
                        value="Record Voice",
                        command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame,
                        text="Load File",
                        variable=self.mode_var,
                        value="Load File",
                        command=self.on_mode_change).pack(anchor=tk.W)

        # Directory selection
        dir_frame = ttk.LabelFrame(self.left_frame, text="Directory", padding="5")
        dir_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(dir_frame, text="Browse Directory", command=self.browse_directory).pack(fill=tk.X, pady=(0, 5))

        # Current directory display
        dir_label = ttk.Label(dir_frame, text="Current Directory:")
        dir_label.pack(anchor=tk.W)

        self.dir_display = ttk.Label(dir_frame,
                                     textvariable=self.current_directory,
                                     wraplength=200,
                                     font=("TkDefaultFont", 8))
        self.dir_display.pack(anchor=tk.W, fill=tk.X)

        # File selection listbox
        file_frame = ttk.LabelFrame(self.left_frame, text="File Selection", padding="5")
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Scrollable listbox
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE, height=8)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        scrollbar.config(command=self.file_listbox.yview)

        # Selected file display
        selected_frame = ttk.Frame(file_frame)
        selected_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(selected_frame, text="Selected:").pack(anchor=tk.W)
        self.selected_label = ttk.Label(selected_frame,
                                        textvariable=self.selected_file,
                                        wraplength=200,
                                        font=("TkDefaultFont", 8),
                                        foreground="blue")
        self.selected_label.pack(anchor=tk.W, fill=tk.X)

        # Filename entry
        filename_frame = ttk.LabelFrame(self.left_frame, text="Filename Entry", padding="5")
        filename_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(filename_frame, text="Enter filename:").pack(anchor=tk.W)
        self.filename_entry = ttk.Entry(filename_frame, textvariable=self.filename_var)
        self.filename_entry.pack(fill=tk.X, pady=(2, 0))

        # Action buttons
        button_frame = ttk.Frame(self.left_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Process", command=self.process_action).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Clear Plots", command=self.clear_plots).pack(fill=tk.X)

    def create_right_panel(self):
        """Create matplotlib subplots in the right panel"""
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(6, 6))
        self.fig.suptitle("Audio Analysis Plots")

        # Configure subplots
        self.ax1.set_title("Time Domain")
        self.ax1.set_xlabel("Time (ms)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(0, 50)

        self.ax2.set_title("Frequency Domain")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Magnitude")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(0, MAX_FREQ)

        self.ax3.set_title('Formant Analysis (F0, F1, F2, F3, F4)')
        self.ax3.set_xlabel("Frequency (Hz)")
        self.ax3.set_ylabel("Magnitude (dB)")
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_xlim(0, MAX_FREQ)

        plt.tight_layout()

        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar
        toolbar_frame = ttk.Frame(self.right_frame)
        toolbar_frame.pack(fill=tk.X)

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def browse_directory(self):
        """Open directory browser and update file list"""
        directory = filedialog.askdirectory(initialdir=self.current_directory.get())
        if directory:
            self.current_directory.set(directory)
            self.update_file_list()

    def update_file_list(self):
        """Update the file listbox with files from current directory"""
        self.file_listbox.delete(0, tk.END)

        try:
            directory = self.current_directory.get()
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory)
                         if os.path.isfile(os.path.join(directory, f))]
                files.sort()

                for file in files:
                    if file[-4:].upper() == ".WAV":
                        self.file_listbox.insert(tk.END, file)
        except Exception as e:
            messagebox.showerror("Error", f"Error reading directory: {str(e)}")

    def on_file_select(self, event):
        """Handle file selection from listbox"""
        selection = self.file_listbox.curselection()
        if selection:
            filename = self.file_listbox.get(selection[0])
            self.selected_file.set(filename)
            # Auto-populate filename entry WITH the extension
            self.filename_var.set(filename)

    def on_mode_change(self):
        """Handle mode change between Record Voice and Load File"""
        mode = self.mode_var.get()
        if mode == "Record Voice":
            self.filename_var.set("TEMP.WAV")
        elif mode == "Load File":
            self.filename_var.set("TEMP.WAV")

    def process_action(self):
        """Main processing function"""
        mode = self.mode_var.get()

        if mode == "Record Voice":
            self.process_record_voice()
        else:  # Load File
            self.process_load_file()

    def process_record_voice(self):
        """Process voice recording"""

        filename = self.filename_var.get()
        if not filename:
            messagebox.showwarning("Warning", "Please enter a filename")
            return
        self.time_length = REC_TIME
        self.n_samples = int(RATE * REC_TIME)
        audio_data = GetFormants.record_voice(filename, self.time_length, self.sample_rate)
        n_samples = len(audio_data)

        t = np.linspace(0, self.n_samples, self.n_samples) / self.sample_rate

        self.plot_audio_data(t, audio_data, f"Recorded: {filename}")
        # messagebox.showinfo("Info", f"Voice recording processed: {filename}")

    def process_load_file(self):
        """Process loaded file"""
        selected = self.selected_file.get()
        if not selected:
            messagebox.showwarning("Warning", "Please select a file")
            return

        filepath = os.path.join(self.current_directory.get(), selected)

        self.sample_rate, audio_data = GetFormants.load_audio(selected)

        number_of_channels = audio_data.shape
        self.n_samples = len(audio_data)
        self.time_length = self.n_samples / self.sample_rate
        t = np.linspace(0, self.n_samples, self.n_samples) / self.sample_rate
        self.plot_audio_data(t, audio_data, f"Loaded: {selected}")
        # messagebox.showinfo("Info", f"File processed: {selected}")

    def plot_audio_data(self, time_data, audio_data, title_suffix=""):
        """Plot audio data in both time and frequency domains and display formants"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        formants, formant_bandwidth = GetFormants.find_formants(audio_data, self.num_formants, self.sample_rate)

        # Time domain plot
        self.ax1.plot(time_data * 1000, audio_data, 'b-', linewidth=1)
        self.ax1.set_title(f"Time Domain - {title_suffix}")
        self.ax1.set_xlabel("Time (ms)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(0, 50)
        self.canvas.draw()

        # Plot 2: FFT Magnitude Spectrum
        fft = np.fft.fft(audio_data)
        # returned values are already masked to remove negative freqs and values
        freqs, magnitude = GetFormants.compute_fft(audio_data, self.sample_rate)

        # Plot only positive frequencies
        self.ax2.plot(freqs, magnitude, 'r-', linewidth=1)
        self.ax2.set_title(f"Frequency Domain - {title_suffix}")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Magnitude")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(0, MAX_FREQ)
        self.canvas.draw()

        # Plot 3: FFT magnitude in dB with Formants and Pitch
        freqs_detailed = np.linspace(0, MAX_FREQ, 1000)
        # Interpolate magnitude spectrum for smoother display
        #mask = freqs > 0
        #magnitude_interp = np.interp(freqs_detailed, freqs[mask], magnitude[mask])
        #mask = freqs > 0
        magnitude_interp = np.interp(freqs_detailed, freqs, magnitude)
        magnitude_db = 20 * np.log10(magnitude_interp + 1e-10)

        self.ax3.plot(freqs_detailed, magnitude_db, 'b-', alpha=0.7)
        self.ax3.set_title('Formant Analysis (F0, F1, F2, F3, F4)')
        self.ax3.set_xlabel('Frequency (Hz)')
        self.ax3.set_ylabel('Magnitude (dB)')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_xlim(0, MAX_FREQ)
        self.canvas.draw()

        # Mark formants
        formant_labels = ['F0', 'F1', 'F2', 'F3', 'F4']
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        if DISP_BANDWIDTHS:
            for i, (formant, bw, label, color) in enumerate(zip(formants, formant_bandwidth, formant_labels, colors)):
                if formant > 0:  # Only plot if formant was found
                    if i == 0:  # Pitch
                        self.ax3.axvline(x=formant, color=color, linestyle='-', linewidth=2,
                                         label=f'{label}: {formant:.0f} Hz')
                    else:
                        self.ax3.axvline(x=formant, color=color, linestyle='-', linewidth=2,
                                         label=f'{label}: {formant:.0f} +/- {bw:.0f} Hz')
                        self.ax3.axvline(x=formant + bw, color=color, linestyle='--', linewidth=2, alpha=0.4)
                        self.ax3.axvline(x=formant - bw, color=color, linestyle='--', linewidth=2, alpha=0.4)
        else:
            for i, (formant, bw, label, color) in enumerate(zip(formants, formant_bandwidth, formant_labels, colors)):
                if formant > 0:  # Only plot if formant was found
                    if i == 0:  # Pitch
                        self.ax3.axvline(x=formant, color=color, linestyle='-', linewidth=2,
                                         label=f'{label}: {formant:.0f} Hz')
                    else:
                        self.ax3.axvline(x=formant, color=color, linestyle='-', linewidth=2,
                                         label=f'{label}: {formant:.0f} +/- {bw:.0f} Hz')

        self.ax3.legend()
        self.canvas.draw()
        plt.legend(loc='upper right')

        plt.tight_layout()
        self.canvas.draw()
        self.canvas.draw()

    def clear_plots(self):
        """Clear both subplot areas"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        """Set up axes and titles"""
        self.ax1.set_title("Time Domain")
        self.ax1.set_xlabel("Time (ms)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(0, 50)

        self.ax2.set_title("Frequency Domain")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Magnitude")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(0, MAX_FREQ)

        self.ax3.set_title('Formant Analysis (F0, F1, F2, F3, F4)')
        self.ax3.set_xlabel('Frequency (Hz)')
        self.ax3.set_ylabel('Magnitude (dB)')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_xlim(0, MAX_FREQ)

        self.canvas.draw()


def main():
    root = tk.Tk()
    app = GetFormantsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
