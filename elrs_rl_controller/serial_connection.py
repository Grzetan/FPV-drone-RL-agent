import serial
import serial.tools.list_ports
import time
from queue import Queue


class SerialConnection:
    """
    Zarządza połączeniem szeregowym z urządzeniem, np. nadajnikiem ELRS.
    Automatycznie wyszukuje port, obsługuje połączenie i komunikację.
    """

    def __init__(self, baudrate=400000, vid_pid="10C4:EA60", description="CP210x"):
        self.port = None
        self.baudrate = baudrate
        self.vid_pid = vid_pid
        self.description = description
        self.serial_port = None
        self.read_queue = Queue()
        self._read_thread = None
        self._reading = False
        self.find_port()

    def find_port(self):
        """Automatycznie znajduje port COM na podstawie VID:PID lub opisu."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if self.vid_pid in port.hwid or self.description in port.description:
                self.port = port.device
                print(f"Znaleziono nadajnik ELRS na porcie: {self.port}")
                return
        print("Nie znaleziono nadajnika ELRS. Sprawdź połączenie i sterowniki.")

    def connect(self):
        """
        Nawiązuje połączenie szeregowe, próbując ponownie co 2 sekundy w razie niepowodzenia.
        Metoda jest blokująca do momentu nawiązania połączenia.
        """
        while not (self.serial_port and self.serial_port.is_open):
            if not self.port:
                print("Port nie został znaleziony. Skanuję ponownie za 2 sekundy...")
                time.sleep(2)
                self.find_port()
                continue

            try:
                self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)
                print(f"Połączono z {self.port} przy {self.baudrate} baud.")
                self.start_reading()
            except serial.SerialException as e:
                print(f"Błąd podczas otwierania portu {self.port}: {e}. Próba ponowna za 2 sekundy...")
                self.serial_port = None
                time.sleep(2)
        return True

    def disconnect(self):
        """Zamyka połączenie szeregowe."""
        self.stop_reading()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("Połączenie szeregowe zamknięte.")

    def _read_loop(self):
        """Pętla wątku odczytującego dane z portu szeregowego."""
        while self._reading and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    self.read_queue.put(data)
                time.sleep(0.001)  # Krótka pauza, aby nie obciążać CPU
            except serial.SerialException:
                print("Błąd odczytu z portu szeregowego. Zamykanie połączenia.")
                self.disconnect()
                break

    def start_reading(self):
        """Uruchamia wątek odczytujący."""
        self._reading = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    def stop_reading(self):
        """Zatrzymuje wątek odczytujący."""
        self._reading = False
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join()

    def write(self, data: bytes):
        """Wysyła dane przez port szeregowy."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write(data)

    def read(self, num_bytes: int) -> bytes:
        """Odczytuje dane z portu szeregowego."""
        if self.serial_port and self.serial_port.is_open:
            return self.serial_port.read(num_bytes)
        return b""
