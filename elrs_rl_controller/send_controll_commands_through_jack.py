import numpy as np
import sounddevice as sd
import time

TOTAL_CHANNELS = 8
FRAME_LENGTH_MS = 22.5
SEPARATOR_PULSE_US = 300

# To do konfiguracji
channel_values_us = [
    1500,  # CH1: Roll (Aileron)   - 0%
    1500,  # CH2: Pitch (Elevator) - 0%
    1000,  # CH3: Throttle        - 0% (Minimum)
    2000,  # CH4: Yaw (Rudder)     - 100%
    1500,  # CH5: Aux 1          - 0%
    1500,  # CH6: Aux 2          - 0%
    1500,  # CH7: Aux 3          - 0%
    1500,  # CH8: Aux 4          - 0%
]

SAMPLE_RATE = 48000
AMPLITUDE = 0.8


ppm_frame_wave = None


def generate_ppm_frame():
    """Generates one complete PPM frame as a square wave."""
    global ppm_frame_wave

    frame_parts = []
    total_frame_time_us = 0

    # Convert timings from microseconds (us) to number of samples
    us_to_samples = lambda us: int(us * SAMPLE_RATE / 1_000_000)

    separator_samples = us_to_samples(SEPARATOR_PULSE_US)
    low_pulse = np.full(separator_samples, -AMPLITUDE, dtype=np.float32)

    # Generate pulses for each channel
    for i in range(TOTAL_CHANNELS):
        value_us = channel_values_us[i]

        # Ensure channel values are within a safe range
        value_us = np.clip(value_us, 900, 2100)

        channel_samples = us_to_samples(value_us)
        high_pulse = np.full(channel_samples, AMPLITUDE, dtype=np.float32)

        frame_parts.append(high_pulse)
        frame_parts.append(low_pulse)

        total_frame_time_us += value_us + SEPARATOR_PULSE_US

    # Calculate the sync gap to fill the rest of the frame
    frame_length_us = FRAME_LENGTH_MS * 1000
    sync_gap_us = frame_length_us - total_frame_time_us

    if sync_gap_us < 0:
        print("Error: Channel pulse widths exceed total frame length!")
        sync_gap_us = 0

    sync_gap_samples = us_to_samples(sync_gap_us)
    sync_pulse = np.full(sync_gap_samples, -AMPLITUDE, dtype=np.float32)
    frame_parts.append(sync_pulse)

    # Combine all parts into a single array
    ppm_frame_wave = np.concatenate(frame_parts)
    print("PPM frame generated successfully.")
    print(f"Total frame duration: {len(ppm_frame_wave) / SAMPLE_RATE * 1000:.2f} ms")


# This callback function is called by the sounddevice library to get more audio data
position = 0


def audio_callback(outdata, frames, time, status):
    global position
    if status:
        print(status)

    chunk_size = len(outdata)
    remaining_in_frame = len(ppm_frame_wave) - position

    if remaining_in_frame >= chunk_size:
        # We can copy a full chunk from the current frame
        outdata[:] = ppm_frame_wave[position : position + chunk_size].reshape(-1, 1)
        position += chunk_size
    else:
        # We've reached the end of the frame, need to loop back
        part1 = ppm_frame_wave[position:]
        part2_size = chunk_size - len(part1)
        part2 = ppm_frame_wave[:part2_size]

        outdata[:] = np.concatenate((part1, part2)).reshape(-1, 1)
        position = part2_size


# --- Main Program ---
if __name__ == "__main__":
    print("Generating PPM waveform...")
    generate_ppm_frame()

    print("\n--- Starting PPM Signal Generation ---")
    print("Connect your PC's audio output to the radio's trainer port.")
    print("Ensure your radio is in 'Master / Jack' trainer mode.")
    print("Verify signals on the radio's channel monitor before connecting a model.")
    print("\nPress Ctrl+C to stop the signal.")

    try:
        # Create and start the audio stream
        with sd.OutputStream(
            channels=1, samplerate=SAMPLE_RATE, callback=audio_callback
        ):
            while True:
                # The callback runs in the background, just keep the script alive
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping PPM signal.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have a valid audio output device.")
