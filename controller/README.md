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


### Main controller app

1. The main process should work in `uv`, it should be run using `uv run main.py`
2. There should be a `config.toml` file with serial ports configuration, camera id, model path, etc...
3. The main process's loop should be controlled by telemetry data, so every time the `attitude` frame comes in
we should run the camera algorithms, run the model and send back the controlls.
4. The controller should have 3 main components, each in different file: `main`, `camera`, `crsf_parser`.
5. The `main` should be the entrypoint and it should use a function from `crsf_parser` which connects to serial port,
and `yields` new parsed attitude frames. (Use python generators). Then it should use the camera module which takes the current
camera frame and detects 4 corners of the refrence paper sheet. After that in should construct the observation (last X actions, last X reference points, attitue, angular velocity). 
This will change so create a seperate function for this that can be easily modified. With this observation it should invoke the model using something code
similar to `test_hover.py`. When in should take the output and pass it to the `crsf` module for it to be sent back to the drone using code similar to `send_controll_commands.py`
6. The whole app should use detailed `logging` with timestamps, file, line etc.
7. Logic MUST be seperated clearly, don't just take the sample code and paste it in, create smaller functions so no function is longer then `50` lines.
8. There should be an option enabled/disabled in config to preview the camera with reference points live.
