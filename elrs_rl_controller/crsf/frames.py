import struct

# Adresy urządzeń
ALL_ENDPOINT = 0x00
LUA_ENDPOINT = 0xEF
HANDSET_ENDPOINT = 0xEA
MODULE_ENDPOINT = 0xEE
FLIGHT_CONTROLLER_ENDPOINT = 0xC8

# Typy ramek
BATTERY_SENSOR_FRAME = 0x08
LINK_STATISTICS_FRAME = 0x14
CHANNELS_FRAME = 0x16
COMMAND_FRAME = 0x32
PING_DEVICES_FRAME = 0x28
PARAMETER_SETTINGS_READ_FRAME = 0x2C
PARAMETER_SETTINGS_WRITE_FRAME = 0x2D

SUBCOMMAND_FRAME = 0x10
CMD_MODEL_SELECT_FRAME = 0x05

UART_SYNC_FRAME = 0xC8
OPENTX_SYNC_FRAME = 0xC8


class TelemetryData:
    """Prosta klasa do przechowywania danych telemetrycznych."""
    def __init__(self, frame_type, data):
        self.type = frame_type
        self.data = data



def crc8_d5(data: bytes) -> int:
    """
    Oblicza sumę kontrolną CRC-8 z wielomianem 0xD5, używaną w CRSF.
    """
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0xD5
            else:
                crc <<= 1
    return crc & 0xFF

def crc_ba(data: bytes) -> int:
    """
    Oblicza sumę kontrolną używaną w ramkach synchronizacyjnych OpenTX.
    Jest to prosta suma bajtów.
    """
    return sum(data) & 0xFF

def create_model_id_frame(model_id: int) -> bytes:
    """
    Tworzy ramkę do wyboru aktywnego modelu w module TX.
    Odpowiednik `CreateModelIDFrame`.
    """
    payload = bytes([
        COMMAND_FRAME,
        MODULE_ENDPOINT,
        HANDSET_ENDPOINT,
        SUBCOMMAND_FRAME,
        CMD_MODEL_SELECT_FRAME,
        model_id
    ])
    
    crc1 = crc_ba(payload)
    crc2 = crc8_d5(payload + bytes([crc1]))
    
    frame_data = bytes([UART_SYNC_FRAME, len(payload) + 2]) + payload + bytes([crc1, crc2])
    return frame_data

def create_ping_devices_frame() -> bytes:
    """
    Tworzy ramkę do odpytania (ping) o dostępne urządzenia w sieci CRSF.
    Odpowiednik `CreatePingDevicesFrame`.
    """
    payload = bytes([
        PING_DEVICES_FRAME,
        ALL_ENDPOINT,
        LUA_ENDPOINT
    ])
    
    crc1 = crc_ba(payload)
    crc2 = crc8_d5(payload + bytes([crc1]))
    
    frame_data = bytes([UART_SYNC_FRAME, len(payload) + 2]) + payload + bytes([crc1, crc2])
    return frame_data

def create_parameter_settings_read_frame(device_id: int, field_id: int, chunk_id: int) -> bytes:
    """
    Tworzy ramkę z żądaniem odczytu konkretnego ustawienia z urządzenia.
    Odpowiednik `CreateParameterSettingsReadFrame`.
    """
    payload = bytes([
        PARAMETER_SETTINGS_READ_FRAME,
        device_id,
        LUA_ENDPOINT,
        field_id,
        chunk_id
    ])
    
    crc1 = crc_ba(payload)
    crc2 = crc8_d5(payload + bytes([crc1]))
    
    frame_data = bytes([UART_SYNC_FRAME, len(payload) + 2]) + payload + bytes([crc1, crc2])
    return frame_data

def create_parameter_setting_write_frame_uint8(device_id: int, field_id: int, field_value: int) -> bytes:
    """
    Tworzy ramkę do zapisu 8-bitowej wartości ustawienia na urządzeniu.
    Odpowiednik `CreateParameterSettingWriteFrameUint8`.
    """
    payload = bytes([
        PARAMETER_SETTINGS_WRITE_FRAME,
        device_id,
        LUA_ENDPOINT,
        field_id,
        field_value
    ])
    
    crc1 = crc_ba(payload)
    crc2 = crc8_d5(payload + bytes([crc1]))
    
    frame_data = bytes([UART_SYNC_FRAME, len(payload) + 2]) + payload + bytes([crc1, crc2])
    return frame_data

def create_parameter_setting_write_frame_uint16(device_id: int, field_id: int, field_value: int) -> bytes:
    """
    Tworzy ramkę do zapisu 16-bitowej wartości ustawienia na urządzeniu.
    Odpowiednik `CreateParameterSettingWriteFrameUint16`.
    """
    payload = bytes([
        PARAMETER_SETTINGS_WRITE_FRAME,
        device_id,
        LUA_ENDPOINT,
        field_id,
        (field_value >> 8) & 0xFF,
        field_value & 0xFF
    ])
    
    crc1 = crc_ba(payload)
    crc2 = crc8_d5(payload + bytes([crc1]))
    
    frame_data = bytes([UART_SYNC_FRAME, len(payload) + 2]) + payload + bytes([crc1, crc2])
    return frame_data

def pack_channels(channels: list[int]) -> bytes:
    """
    Pakuje 16 kanałów RC (każdy 11-bitowy) do 22 bajtów danych.
    Odpowiednik `PackChannels`.
    """
    if len(channels) != 16:
        raise ValueError("Lista kanałów musi zawierać 16 wartości.")

    payload = bytearray(23)
    payload[0] = CHANNELS_FRAME
    
    bits = 0
    bits_available = 0
    
    packed_bytes = bytearray(22)
    
    for i in range(16):
        val = channels[i] & 0x7FF
        bits |= val << bits_available
        bits_available += 11
        
        while bits_available >= 8:
            index = (i * 11) // 8
            if index < 22:
                packed_bytes[index] = bits & 0xFF
            bits >>= 8
            bits_available -= 8

    payload[1:] = packed_bytes
    
    frame = bytearray()
    frame.append(MODULE_ENDPOINT)
    frame.append(len(payload) + 1)
    frame.extend(payload)
    frame.append(crc8_d5(payload)) 
    
    return bytes(frame)

def parse_frame(buffer: bytearray) -> (TelemetryData, int):
    """
    Parsuje bufor w poszukiwaniu ramki CRSF.
    Zwraca obiekt TelemetryData i długość przetworzonej ramki, lub (None, 0).
    """
    # Szukamy początku ramki (adresu urządzenia)
    sync_index = -1
    for i in range(len(buffer)):
        if buffer[i] in [FLIGHT_CONTROLLER_ENDPOINT, MODULE_ENDPOINT, HANDSET_ENDPOINT]:
            sync_index = i
            break

    if sync_index == -1:
        return None, 0 # Nie znaleziono początku ramki

    # Sprawdzamy, czy mamy wystarczająco danych na nagłówek (długość)
    if len(buffer) < sync_index + 2:
        return None, sync_index # Za mało danych, ale usuń śmieci przed sync

    frame_len = buffer[sync_index + 1]
    # Długość ramki to długość payload + 1 bajt na typ ramki + 1 bajt na CRC
    # Całkowita długość do odczytania to adres + długość + payload
    total_frame_len = frame_len + 2 

    if len(buffer) < sync_index + total_frame_len:
        return None, sync_index # Niekompletna ramka, ale usuń śmieci przed sync

    frame_data = buffer[sync_index : sync_index + total_frame_len]
    payload = frame_data[2:-1] # Dane między nagłówkiem a CRC
    crc_received = frame_data[-1]
    crc_calculated = crc8_d5(payload)

    if crc_received == crc_calculated:
        frame_type = payload[0]
        data_payload = payload[1:]
        telemetry = TelemetryData(frame_type, data_payload)
        return telemetry, total_frame_len + sync_index # Zwróć dane i całkowitą przetworzoną długość
    else:
        # Błąd CRC, usuwamy tylko bajt synchronizacyjny i próbujemy dalej
        return None, sync_index + 1
