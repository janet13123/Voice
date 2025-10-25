# Voice Project

**A small collection of programs to compare, measure, and  learn about various voice parameters.**

# Programs #

GetFormants.py

    - non-GUI interface

    - Takes a snapshot of the voice and saves to TEMP.wav file

    - displays an FFT of the audio with linear magnitude at each frequency

    - also displays an FFT of the audio magnitude in dB and has lines for 
      the pitch, and up to four of the formants.

GetFormantsGUI.py

    - uses Tkinter for the GUI interface

    - Also takes a snapshot of the voice from the microphone.

    - allows you to name the file, as long as it's a .wav file

    - allows you to load an existing wave file also

    - Displays a portion of the time based signal

    - Displays the linear magnitude of the FFT of the audio signal

    - Displays the pitch and up to four formants of the voice with the
      magnitude in dB of the FFT. Also estimates the formant bandwidths.

VoiceView.py

    - non-GUI interface

    - This displays the voice plots in close to real-time.

    - The linear magnitude of the voice's FFT is displayed in the top plot.

    - The bottom plot displays the dB magnitude of the voice FFT with the
      pitch and first four formants.

VoiceViewGUI.py

    - uses Tkinter for the GUI interface

    - This displays the voice plots in close to real-time.

    - The linear magnitude of the voice's FFT is displayed in the top plot.

    - The bottom plot displays the dB magnitude of the voice FFT with the
      pitch and first four formants.

    - Also, an experimental measurement of Closed Quotient is implemented by 
      examining the spectral rolloff of the FFT peaks

    - An experimental measure of breathiness if implemented also but it needs
      some more work.

    - The maximum frequency to display can be changed by using the drop down
      box 'Freq Limit (Hz)' but the display must be paused first by clicking
      the Pause button. After setting the frequency limit, press 'Start' to 
      restart the display.

    - Also, to try the 'Pitch Detection', the display must be paused and then
      restarted.
    
## Installation

### Requirements
- Python 3.12 or higher
- a microphone

### Setup

1. **Download a .zip of the files**

    go to
 
    https://github.com/janet13123/Voice
    
    click the green CODE button 
    
    select 'Download ZIP'
    
    install in an empty directory

2. **Install Python**

3. **Install the dependencies**

    pip install -r requirements.txt (has not been tested yet)

3. **Run one the applications:**

    python GetFormants.py
 
    or
 
    python GetFormantsGUI.py
 
    or
 
    python VoiceView.py
 
    or
 
    python VoiceViewGUI.py
