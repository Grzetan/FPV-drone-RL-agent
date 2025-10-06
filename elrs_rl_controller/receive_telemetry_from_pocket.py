import serial
import struct
import math

# --- Configuration ---
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 460800

# CRSF Constants
SYNC_BYTE = 0xEA
ATTITUDE_FRAME_TYPE = 0x1E


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        print("Listening for Attitude frames... Press Ctrl+C to exit.")
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}. {e}")
        return

    try:
        while True:
            # Hunt for the CRSF Sync Byte
            byte = ser.read(1)
            if not byte or byte[0] != SYNC_BYTE:
                continue

            # Read the frame header (Length and Type)
            header = ser.read(2)
            if len(header) < 2:
                continue
            frame_length, frame_type = header

            # Filter for Attitude frames and verify correct length
            if frame_type == ATTITUDE_FRAME_TYPE and frame_length == 8:
                payload_and_crc = ser.read(7)
                if len(payload_and_crc) < 7:
                    continue  # Incomplete frame

                payload = payload_and_crc[:6]

                # Unpack the payload into three raw signed 16-bit integers
                pitch_raw, roll_raw, yaw_raw = struct.unpack("<hhh", payload)

                # Convert to degrees based on the spec (value is radians * 10000)
                pitch_deg = math.degrees(pitch_raw / 10000.0)
                roll_deg = math.degrees(roll_raw / 10000.0)
                yaw_deg = math.degrees(yaw_raw / 10000.0)

                # Print the results as separate lines
                print(f"Pitch: {pitch_deg:.1f}")
                print(f"Roll: {roll_deg:.1f}")
                print(f"Yaw: {yaw_deg:.1f}")
                print("---")  # Separator for readability

            else:
                # Discard any other frame type
                ser.read(frame_length - 1)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if ser.is_open:
            ser.close()
            print("Serial port closed.")


if __name__ == "__main__":
    main()
