### Reading telemetry data from RadioMaster
1. Connect binded radio to PC
2. Pick USB Serial mode on radio master
3. Check in `dmesg` what serial port was assigned
4. Put it into `SERIAL_PORT` in `receive_telemetry_from_pocket.py`
5. Check the Baud rate of the radio by connecting to RX wifi and going into `10.0.0.1`
6. Controll the frequency of attitude packets by going in expresslrs setting on radio, setting the  packet rate (prefferably to 500hz) and Telem ratio to 1:32. This means 1 telemetry packet per 32 channel packets. Lower is faster (1:8 is faster then 1:64).
7. Go into hardware setting on radio and set the USB-VCP to `Telem Mirror`

### Sending stick commands from PC to radiomaster
1. MDL -> Setup -> Trainer mode -> set to Master/Jack
2. Sys -> Global Functions -> Add new function -> ON (or some switch to enable/disable trainer mode), Trainer, Sticks. Make sure to enable this function.
3. Connect the PC with the radio master using jack
4. Change the audio output on PC to the newly connected jack.
5. Run the `send_controll_commands_through_jack.py` and the channels on the radiomaster should be changed and the original sticks should not work