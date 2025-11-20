import numpy as np
import sounddevice as sd
import time
import threading

TOTAL_CHANNELS = 8
FRAME_LENGTH_MS = 22.5
SEPARATOR_PULSE_US = 300
SAMPLE_RATE = 48000
AMPLITUDE = 0.8
INVERT_POLARITY = False

channel_values_us = [1500] * TOTAL_CHANNELS

ppm_frame_wave = None
position = 0
lock = threading.Lock()

def generate_ppm_frame():
    global ppm_frame_wave
    
    frame_parts = []
    total_frame_time_us = 0
    
    us_to_samples = lambda us: int(us * SAMPLE_RATE / 1_000_000)
    
    if INVERT_POLARITY:
        lvl_pulse = -AMPLITUDE
        lvl_wait  = AMPLITUDE
    else:
        lvl_pulse = AMPLITUDE
        lvl_wait  = -AMPLITUDE

    sep_samples = us_to_samples(SEPARATOR_PULSE_US)
    pulse_wave = np.full(sep_samples, lvl_pulse, dtype=np.float32)

    for i in range(TOTAL_CHANNELS):
        val = np.clip(channel_values_us[i], 700, 2300)
        
        wait_time_us = val - SEPARATOR_PULSE_US
        wait_samples = us_to_samples(wait_time_us)
        wait_wave = np.full(wait_samples, lvl_wait, dtype=np.float32)
        
        frame_parts.append(pulse_wave)
        frame_parts.append(wait_wave)
        
        total_frame_time_us += val

    frame_parts.append(pulse_wave)
    total_frame_time_us += SEPARATOR_PULSE_US
    
    frame_len_us = FRAME_LENGTH_MS * 1000
    sync_gap_us = frame_len_us - total_frame_time_us
    
    if sync_gap_us < 3000:
        sync_gap_us = 4000 
        
    sync_samples = us_to_samples(sync_gap_us)
    sync_wave = np.full(sync_samples, lvl_wait, dtype=np.float32)
    
    frame_parts.append(sync_wave)

    new_wave = np.concatenate(frame_parts).reshape(-1, 1)

    with lock:
        ppm_frame_wave = new_wave

def audio_callback(outdata, frames, time_info, status):
    global position, ppm_frame_wave
    if status:
        print(status)

    with lock:
        if ppm_frame_wave is None:
            outdata.fill(0)
            return

        chunk_size = len(outdata)
        wave_len = len(ppm_frame_wave)
        
        if position >= wave_len:
            position %= wave_len

        end_pos = position + chunk_size
        
        if end_pos <= wave_len:
            outdata[:] = ppm_frame_wave[position:end_pos]
            position += chunk_size
        else:
            part1_len = wave_len - position
            outdata[:part1_len] = ppm_frame_wave[position:]
            
            part2_len = chunk_size - part1_len
            outdata[part1_len:] = ppm_frame_wave[:part2_len]
            
            position = part2_len
            
        if position >= wave_len:
            position = 0

if __name__ == "__main__":
    generate_ppm_frame()
    
    print("\n--- Starting PPM Signal Generation ---")
    print("Connect your PC's audio output to the radio's trainer port.")
    print("Ensure your radio is in 'Master / Jack' trainer mode.")
    print("Verify signals on the radio's channel monitor before connecting a model.")
    print("\nPress Ctrl+C to stop the signal.")
    print("--------------------------------------\n")
    
    try:
        with sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
            
            print("Calibrate input signals on radio:")
            print("SYS -> TRAINER -> Long press 'Cal'")
            print("If you have done this, press Enter...")
            input("")
            print("Starting simulation...")
            
            current_channel_index = 0
            direction = 20
            min_val, max_val = 900, 2100
            
            while True:
                # Zresetuj nieaktywne kanały do wartości środkowej
                for i in range(4):
                    if i != current_channel_index:
                        channel_values_us[i] = 1500
                
                # Zaktualizuj wartość aktywnego kanału
                channel_values_us[current_channel_index] += direction
                
                # Sprawdź granice i zmień kierunek lub kanał
                if channel_values_us[current_channel_index] >= max_val:
                    channel_values_us[current_channel_index] = max_val
                    direction = -20
                elif channel_values_us[current_channel_index] <= min_val:
                    channel_values_us[current_channel_index] = min_val
                    direction = 20
                    # Koniec pełnego cyklu (góra-dół), przejdź do następnego kanału
                    channel_values_us[current_channel_index] = 1500 # Zresetuj stary kanał
                    current_channel_index = (current_channel_index + 1) % 4
                
                generate_ppm_frame()
                
                print(f"Testing CH{current_channel_index + 1}: {channel_values_us[current_channel_index]:4d}us", end='     \r')
                
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")