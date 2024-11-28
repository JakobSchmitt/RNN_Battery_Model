# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:27:10 2024

@author: s8940173
"""


# Load the checkpoint
checkpoint_path = r"C:\Users\s8940173\Dokumente\GitHub\RNN_Battery_Model\checkpoints\2024-11-28\avid-sea-233\weights\best_model\model.ckpt"

import torch
import pickle
import os
from pathlib import WindowsPath, PosixPath

# Custom unpickler to fix persistent ID issues
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map PosixPath to WindowsPath on Windows systems
        if module == "pathlib" and name == "PosixPath":
            return WindowsPath
        return super().find_class(module, name)
    
    def persistent_load(self, pid):
        """
        Custom handling of persistent load. Forcibly convert non-ASCII persistent IDs into a valid form.
        """
        print(f"Debug: Persistent ID encountered: {pid}")
        
        # If it's a tuple, handle it normally
        if isinstance(pid, tuple):
            return torch._utils._persistent_load(pid)
        
        # Handle bytes (possible non-ASCII IDs)
        if isinstance(pid, bytes):
            try:
                return pid.decode("ascii")  # Try decoding to ASCII
            except UnicodeDecodeError:
                # If decoding fails, try converting to string (ensuring it's safe for Python)
                print(f"Error decoding PID: {pid}. Converting to string.")
                return str(pid)  # Fallback to string conversion
        
        # If it's not a recognized type, raise an error
        raise pickle.UnpicklingError(f"Unsupported persistent ID format: {pid}")

def custom_torch_load(checkpoint_path, map_location=None):
    """
    Custom loader to fix issues with loading PyTorch models that have non-ASCII persistent IDs.
    """
    with open(checkpoint_path, "rb") as f:
        unpickler = CustomUnpickler(f)
        checkpoint = unpickler.load()

    # If needed, map the checkpoint to the correct device
    if map_location:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    return checkpoint

def load_checkpoint_with_path_fix(checkpoint_path):
    """
    Load the checkpoint while fixing potential path issues between Unix-based and Windows systems.
    """
    # Convert the checkpoint path to a proper format if necessary
    checkpoint_path = str(checkpoint_path)  # Make sure the path is a string
    if os.name == 'nt':  # Check if we're on a Windows system
        checkpoint_path = checkpoint_path.replace('/', '\\')  # Fix Unix-style paths to Windows style
    
    try:
        checkpoint = custom_torch_load(checkpoint_path, map_location=torch.device('cpu'))
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")



# Attempt to load the checkpoint
load_checkpoint_with_path_fix(checkpoint_path)
