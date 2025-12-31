import serial
import logging
from enum import IntEnum

CRSF_SYNC = 0xC8

class PacketsTypes(IntEnum):
    GPS = 0x02
    VARIO = 0x07
    BATTERY_SENSOR = 0x08
    BARO_ALT = 0x09
    HEARTBEAT = 0x0B
    VIDEO_TRANSMITTER = 0x0F
    LINK_STATISTICS = 0x14
    RC_CHANNELS_PACKED = 0x16
    ATTITUDE = 0x1E
    FLIGHT_MODE = 0x21
    DEVICE_INFO = 0x29
    CONFIG_READ = 0x2C
    CONFIG_WRITE = 0x2D
    RADIO_ID = 0x3A

def crc8_dvb_s2(crc, a) -> int:
    crc = crc ^ a
    for _ in range(8):
        if crc & 0x80:
            crc = (crc << 1) ^ 0xD5
        else:
            crc = crc << 1
    return crc & 0xFF

def crc8_data(data) -> int:
    crc = 0
    for a in data:
        crc = crc8_dvb_s2(crc, a)
    return crc

def crsf_validate_frame(frame) -> bool:
    return crc8_data(frame[2:-1]) == frame[-1]

def parse_attitude_frame(data):
    pitch = int.from_bytes(data[3:5], byteorder='big', signed=True) / 10000.0
    roll = int.from_bytes(data[5:7], byteorder='big', signed=True) / 10000.0
    yaw = int.from_bytes(data[7:9], byteorder='big', signed=True) / 10000.0
    return {'pitch': pitch, 'roll': roll, 'yaw': yaw}

def get_attitude_frames(port: str, baud_rate: int):
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
    except serial.SerialException as e:
        logging.error(f"Failed to open serial port {port}: {e}")
        return

    input_buffer = bytearray()
    with ser:
        while True:
            if ser.in_waiting > 0:
                input_buffer.extend(ser.read(ser.in_waiting))

            while len(input_buffer) > 2:
                expected_len = input_buffer[1] + 2
                if not (4 <= expected_len <= 64):
                    input_buffer.pop(0)
                    continue

                if len(input_buffer) >= expected_len:
                    frame = input_buffer[:expected_len]
                    input_buffer = input_buffer[expected_len:]

                    if not crsf_validate_frame(frame):
                        logging.warning("CRSF frame validation failed.")
                        continue
                    
                    packet_type = frame[2]
                    if packet_type == PacketsTypes.ATTITUDE:
                        yield parse_attitude_frame(frame)
                else:
                    break

def pack_crsf_channels(channels):
    if len(channels) != 16:
        raise ValueError('CRSF must have 16 channels')
    
    result = bytearray()
    dest_shift = 0
    new_val = 0
    for ch in channels:
        new_val |= (ch << dest_shift) & 0xff
        result.append(new_val)

        src_bits_left = 11 - 8 + dest_shift
        new_val = ch >> (11 - src_bits_left)
        
        if src_bits_left >= 8:
            result.append(new_val & 0xff)
            new_val >>= 8
            src_bits_left -= 8

        dest_shift = src_bits_left
        
    return result

def create_channels_packet(channels):
    payload = pack_crsf_channels(channels)
    packet_len = len(payload) + 2
    result = bytearray([CRSF_SYNC, packet_len, PacketsTypes.RC_CHANNELS_PACKED])
    result += payload
    result.append(crc8_data(result[2:]))
    return result

def send_control_data(ser: serial.Serial, channels):
    packet = create_channels_packet(channels)
    ser.write(packet)
