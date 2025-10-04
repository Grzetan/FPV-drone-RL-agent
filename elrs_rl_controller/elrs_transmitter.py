import time
import threading
from crsf import frames
from serial_connection import SerialConnection


class ELRSTransmitter:
    """
    API do komunikacji z nadajnikiem ELRS.
    Używa SerialConnection do wysyłania ramek CRSF.
    """

    def __init__(self, connection: SerialConnection):
        self.conn = connection
        self._channel_data = [172] * 16  # Domyślne wartości minimalne dla kanałów
        self._channel_data[0] = 992  # Roll center
        self._channel_data[1] = 992  # Pitch center
        self._channel_data[2] = 172  # Throttle low
        self._channel_data[3] = 992  # Yaw center
        self._running = False
        self._send_thread = None
        self._telemetry_thread = None
        self._telemetry_buffer = bytearray()

        # Dane telemetryczne
        self.uplink_rssi_1 = -130
        self.uplink_rssi_2 = -130
        self.uplink_link_quality = 0
        self.uplink_snr = 0
        self.downlink_rssi = -130
        self.downlink_link_quality = 0
        self.downlink_snr = 0
        self.battery_voltage = 0.0
        self.battery_current = 0.0
        self.battery_capacity_drawn = 0

    def start_sending_channels(self, interval=0.004):
        """Uruchamia wątek, który cyklicznie wysyła pozycje kanałów."""
        if not self.conn.serial_port or not self.conn.serial_port.is_open:
            print("Nie można uruchomić wysyłania - brak aktywnego połączenia.")
            return

        self._running = True
        self._send_thread = threading.Thread(target=self._send_loop, args=(interval,), daemon=True)
        self._send_thread.start()
        print("Uruchomiono cykliczne wysyłanie danych kanałów.")

        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._telemetry_thread.start()
        print("Uruchomiono nasłuchiwanie telemetrii.")

    def stop_sending_channels(self):
        """Zatrzymuje cykliczne wysyłanie pozycji kanałów i odbieranie telemetrii."""
        self._running = False
        if self._send_thread:
            self._send_thread.join()
        if self._telemetry_thread:
            self._telemetry_thread.join()
        print("Zatrzymano cykliczne wysyłanie danych kanałów.")

    def _send_loop(self, interval: float):
        """Pętla wątku wysyłającego."""
        while self._running:
            self.send_rc_channels()
            time.sleep(interval)

    def _telemetry_loop(self):
        """Pętla wątku odbierającego i parsującego telemetrię."""
        while self._running:
            while not self.conn.read_queue.empty():
                self._telemetry_buffer.extend(self.conn.read_queue.get())

            telemetry_data, processed_len = frames.parse_frame(self._telemetry_buffer)
            if telemetry_data:
                self._update_telemetry(telemetry_data)
                self._telemetry_buffer = self._telemetry_buffer[processed_len:]
            elif processed_len > 0: # Usunięcie śmieci z bufora
                self._telemetry_buffer = self._telemetry_buffer[processed_len:]
            else:
                time.sleep(0.001) # Czekaj na więcej danych

    def _update_telemetry(self, telemetry: frames.TelemetryData):
        """Aktualizuje stan telemetrii na podstawie sparsowanej ramki."""
        if telemetry.type == frames.LINK_STATISTICS_FRAME and len(telemetry.data) >= 10:
            self.uplink_rssi_1 = telemetry.data[0] - 130
            self.uplink_rssi_2 = telemetry.data[1] - 130
            self.uplink_link_quality = telemetry.data[2]
            self.uplink_snr = telemetry.data[3]
            self.downlink_rssi = telemetry.data[5] - 130
            self.downlink_link_quality = telemetry.data[7]
            self.downlink_snr = telemetry.data[8]
        elif telemetry.type == frames.BATTERY_SENSOR_FRAME and len(telemetry.data) >= 8:
            voltage_raw = (telemetry.data[0] << 8) | telemetry.data[1]
            current_raw = (telemetry.data[2] << 8) | telemetry.data[3]
            capacity_raw = (telemetry.data[4] << 16) | (telemetry.data[5] << 8) | telemetry.data[6]
            self.battery_voltage = voltage_raw / 10.0
            self.battery_current = current_raw / 10.0
            self.battery_capacity_drawn = capacity_raw

    def set_channel(self, channel_index: int, value: int):
        """Ustawia wartość dla pojedynczego kanału (0-15). Wartość 11-bitowa (0-2047)."""
        if 0 <= channel_index < 16:
            # CRSF używa wartości od 172 (min) do 1811 (max), środek to 992
            self._channel_data[channel_index] = int(value)
        else:
            print(f"Błąd: Nieprawidłowy indeks kanału: {channel_index}")

    def set_channels_from_array(self, values: list[int]):
        """Ustawia wartości dla wszystkich 16 kanałów z listy."""
        if len(values) == 16:
            self._channel_data = values
        else:
            print("Błąd: Lista musi zawierać 16 wartości kanałów.")

    def send_rc_channels(self):
        """Tworzy i wysyła ramkę z aktualnymi pozycjami kanałów."""
        frame = frames.pack_channels(self._channel_data)
        self.conn.write(frame)

    def select_model(self, model_id: int):
        """Wysyła komendę wyboru modelu."""
        frame = frames.create_model_id_frame(model_id)
        self.conn.write(frame)
        print(f"Wysłano żądanie wyboru modelu ID: {model_id}")


if __name__ == "__main__":
    import keyboard
    import argparse

    parser = argparse.ArgumentParser(description="Sterowanie nadajnikiem ELRS przez port szeregowy.")
    parser.add_argument(
        '--baudrate', 
        type=int, 
        default=400000, 
        help='Prędkość portu szeregowego (baud rate). Domyślnie: 400000'
    )
    args = parser.parse_args()

    conn = SerialConnection(baudrate=args.baudrate)
    conn.connect()

    transmitter = ELRSTransmitter(conn)
    transmitter.select_model(0)
    transmitter.start_sending_channels()

    try:
        while True:
            throttle_value = 992 + (819 if keyboard.is_pressed('up') else (-820 if keyboard.is_pressed('down') else 0))
            transmitter.set_channel(2, throttle_value)

            print(
                f"\rLQ: {transmitter.uplink_link_quality}% | RSSI: {transmitter.uplink_rssi_1}dBm | VBat: {transmitter.battery_voltage:.2f}V | Curr: {transmitter.battery_current:.2f}A",
                end=""
            )

            time.sleep(0.05)
            if keyboard.is_pressed('q'):
                break
    finally:
        print("\nZamykanie...")
        transmitter.stop_sending_channels()
        conn.disconnect()
        print("Zakończono.")
