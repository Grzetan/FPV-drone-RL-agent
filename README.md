Valid agent observation:
- attitude (current yaw, pitch, roll)
- angular velocity (calculated from the series of attitudes)
- linear velocity (try without that for now, if RL doesnt worm then write a custom lua script that requests the raw acc data in MSP in CRSF protocol 

To view tensorboard:
tensorboard --logdir logs