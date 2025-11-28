import pygame
import numpy as np

class RadioMasterJoystick:
    """
    RadioMaster Pocket joystick controller for drone control.
    Maps axis values from RadioMaster Pocket to drone controls.
    """
    
    def __init__(self, joystick_index=0):
        """
        Initialize the RadioMaster joystick controller.
        
        Args:
            joystick_index: Index of the joystick device (default: 0)
        """
        pygame.init()
        
        # Check for connected joysticks
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise RuntimeError("No joystick detected. Please connect your RadioMaster Pocket.")
        
        if joystick_index >= joystick_count:
            raise RuntimeError(f"Joystick index {joystick_index} not available. Found {joystick_count} joystick(s).")
        
        # Initialize the joystick
        self.joystick = pygame.joystick.Joystick(joystick_index)
        self.joystick.init()
        
        print(f"Initialized RadioMaster Joystick: {self.joystick.get_name()}")
        print(f"Number of Axes: {self.joystick.get_numaxes()}")
        print(f"Number of Buttons: {self.joystick.get_numbuttons()}")
        
        # Control mappings
        self.axis_mappings = {
            'roll': 0,      # Axis 0: Roll (left/right)
            'pitch': 1,     # Axis 1: Pitch (forward/backward)
            'throttle': 2,  # Axis 2: Throttle (up/down)
            'yaw': 3        # Axis 3: Yaw (rotation)
        }
        
        # Control scaling factors
        self.scale_factors = {
            'roll': 1.0,    # Full range for roll
            'pitch': 1.0,   # Full range for pitch
            'throttle': 1.0, # Full range for throttle
            'yaw': 1.0      # Full range for yaw
        }
        
        # Dead zone to prevent small movements
        self.dead_zone = 0.05
        
    def get_controls(self):
        """
        Get current control values from the RadioMaster Pocket.
        
        Returns:
            dict: Dictionary with control values {'roll': float, 'pitch': float, 'throttle': float, 'yaw': float}
        """
        # Process pygame events
        pygame.event.pump()
        
        controls = {}
        
        for control_name, axis_index in self.axis_mappings.items():
            if axis_index < self.joystick.get_numaxes():
                # Get raw axis value (-1 to 1)
                raw_value = self.joystick.get_axis(axis_index)
                
                # Apply dead zone
                if abs(raw_value) < self.dead_zone:
                    raw_value = 0.0
                
                # Apply scaling
                scaled_value = raw_value * self.scale_factors[control_name]
                
                # Clamp to [-1, 1] range
                controls[control_name] = np.clip(scaled_value, -1.0, 1.0)
            else:
                controls[control_name] = 0.0
        
        return controls
    
    def get_action_array(self):
        """
        Get control values as numpy array for drone control.
        
        Returns:
            np.array: Array with [roll, pitch, yaw, throttle] values in [-1, 1] range
        """
        controls = self.get_controls()
        return np.array([
            controls['roll'],
            controls['pitch'], 
            controls['yaw'],
            controls['throttle']
        ], dtype=np.float32)
    
    def set_scale_factor(self, control_name, scale):
        """
        Set scaling factor for a specific control.
        
        Args:
            control_name: Name of the control ('roll', 'pitch', 'throttle', 'yaw')
            scale: Scaling factor (float)
        """
        if control_name in self.scale_factors:
            self.scale_factors[control_name] = scale
        else:
            raise ValueError(f"Invalid control name: {control_name}")
    
    def set_dead_zone(self, dead_zone):
        """
        Set dead zone for all controls.
        
        Args:
            dead_zone: Dead zone value (0.0 to 1.0)
        """
        self.dead_zone = np.clip(dead_zone, 0.0, 1.0)
    
    def print_status(self):
        """
        Print current control values for debugging.
        """
        controls = self.get_controls()
        print(f"Roll: {controls['roll']:6.3f} | Pitch: {controls['pitch']:6.3f} | "
              f"Throttle: {controls['throttle']:6.3f} | Yaw: {controls['yaw']:6.3f}", end="\r")
    
    def cleanup(self):
        """
        Clean up pygame resources.
        """
        pygame.quit()
