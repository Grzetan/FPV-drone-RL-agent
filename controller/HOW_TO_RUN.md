# Instrukcja Uruchomienia Kontrolera Drona

## 1. Instalacja Zależności

Całe środowisko jest zarządzane przez `uv` w folderze `simulation`.

1.  Przejdź do folderu `simulation`:
    ```bash
    cd simulation
    ```
2.  Zainstaluj zależności z pliku `pyproject.toml`:
    ```bash
    uv pip sync pyproject.toml
    ```

## 2. Konfiguracja

Przed uruchomieniem kontrolera, skonfiguruj odpowiednio plik `controller/config.toml`. Plik ten znajduje się w folderze `controller`.

### Konfiguracja Ogólna
- `[serial]`:
  - `port`: Ustaw port szeregowy, do którego podłączony jest odbiornik telemetryczny (np. `/dev/ttyUSB0` w Linuksie lub `COM3` w Windows).
    - **Jak znaleźć port szeregowy?**
      - **Linux**: Po podłączeniu urządzenia USB, użyj komendy `dmesg | grep tty` w terminalu. Szukaj wpisów takich jak `ttyUSB0` lub `ttyACM0`.
      - **Windows**: Sprawdź w `Menedżerze Urządzeń` (Device Manager) w sekcji `Porty (COM i LPT)`.
  - `baud_rate`: Ustaw szybkość transmisji zgodną z twoim odbiornikiem.
    - **Jak znaleźć Baud Rate?**
      - Ta informacja zazwyczaj jest dostępna w ustawieniach twojego odbiornika (np. w menu ExpressLRS na radiu, lub w dokumentacji urządzenia). Upewnij się, że wartość w `config.toml` jest identyczna.
- `[camera]`:
  - `id`: Podaj ID kamery, której chcesz używać (zazwyczaj `0` lub `1`).
    - **Jak znaleźć ID kamery?**
      - Zazwyczaj, domyślna kamera systemowa ma ID `0`. Możesz spróbować różnych wartości (`0`, `1`, `2` itd.), aby znaleźć właściwą.
      - **Linux**: Możesz użyć `v4l2-ctl --list-devices` (jeśli `v4l2-utils` jest zainstalowane), aby wyświetlić dostępne kamery i ich indeksy.
  - `preview`: Ustaw `true`, jeśli chcesz widzieć podgląd z kamery na żywo.

### Konfiguracja Modelu (Dwa Tryby)

#### A) Tryb Zmockowany (do testów, bez modelu AI)
Ten tryb pozwala na uruchomienie aplikacji bez prawdziwego modelu. Kontroler będzie wysyłał neutralne wartości sterujące. Jest to idealne do testowania połączenia, kamery i przepływu danych.

Aby włączyć ten tryb, ustaw w sekcji `[model]`:
```toml
[model]
use_mock = true
```
W tym trybie wartość `path` jest ignorowana.

#### B) Tryb Rzeczywisty (z modelem AI)
Ten tryb używa wytrenowanej sieci neuronowej do podejmowania decyzji o sterowaniu.

Aby włączyć ten tryb, ustaw w sekcji `[model]`:
```toml
[model]
use_mock = false
path = "sciezka/do/twojego/modelu" # WAŻNE: Podaj poprawną ścieżkę
```
Pamiętaj, aby podać ścieżkę **bez rozszerzenia pliku** (`.zip` lub `.pkl`).

## 3. Podłączenie Sprzętu

1.  **Odbiornik RC**: Podłącz odbiornik do komputera przez port USB. Upewnij się, że jest w trybie `Telem Mirror`.
2.  **Kamera**: Podłącz kamerę do komputera.

## 4. Uruchomienie Kontrolera

Główny skrypt kontrolera należy uruchomić z folderu `simulation`, używając `uv run`. Aplikacja zachowa się inaczej w zależności od konfiguracji w `config.toml`.

```bash
# Upewnij się, że jesteś w folderze /simulation
uv run python ../controller/main.py
```

Po uruchomieniu, w konsoli powinny pojawić się logi informujące o statusie aplikacji. Jeśli podgląd kamery jest włączony, na ekranie pojawi się dodatkowe okno.

Aby zatrzymać kontroler, naciśnij `Ctrl+C` w terminalu.
