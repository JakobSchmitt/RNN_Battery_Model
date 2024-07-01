import torch
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


class SEFDataset(Dataset):
    """
    SEFDataset: A custom PyTorch Dataset implementation
    https://pytorch.org/docs/stable/data.html#map-style-datasets 

    This dataset class applies sliding window techniques to generate encoder and decoder inputs and outputs. 
    It takes a full voltage discharge curve as input and generates samples based on the input parameters
    
    Parameters:
    -----------
    root_dir : Path
        The root directory containing the profiles' data.
    profiles : List[int]
        A list of profile IDs to include in the dataset.
    encoder_input_length : int, (default=20)
        The length of the encoder input sequence.
    decoder_input_length : int, (default=1980)
        The length of the decoder input sequence.
    stride : int, optional (default=16)
        The stride length for the sliding window.
    normalize : bool, optional (default=True)
        Whether to normalize the voltage and current values.
 

    Methods:
    --------
    __len__():
        Returns the total number of samples in the dataset.
    __getitem__(idx: int):
        Retrieves the encoder input, decoder input, and decoder output for the given index.
    """
    def __init__(
        self,
        root_dir: Path,
        profiles: List[int],
        encoder_input_length: int = 20,
        decoder_input_length: int = 1980,
        stride: int = 16,
        normalize: bool = True,
        voltage_min: float = 2.5,
        voltage_max: float = 4.2,
        current_min: float = -2.5,
        current_max: float = 1.0,
    ):
        assert stride < (encoder_input_length + decoder_input_length)
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.current_min = current_min
        self.current_max = current_max
        encoder_input = []
        decoder_input = []
        decoder_output = []
        for profile in profiles:
            curves_dir = sorted((root_dir / str(profile)).iterdir())
            for curve_dir in curves_dir:
                curve = pd.read_csv(curve_dir, usecols=["voltage", "current"])
                assert curve.shape[0] > (encoder_input_length + decoder_input_length)
                assert curve.shape[0] > stride
                if normalize:
                    curve["voltage"] = (curve["voltage"] - self.voltage_min) / (
                        self.voltage_max - self.voltage_min
                    )
                    curve["current"] = (curve["current"] - self.current_min) / (
                        self.current_max - self.current_min
                    )
                curve_sliding_samples = sliding_window_view(
                    curve, (encoder_input_length + decoder_input_length, 2)
                ).squeeze()
                if curve_sliding_samples.shape[0] % stride != 1:
                    curve_samples = torch.tensor(
                        np.concatenate(
                            (
                                curve_sliding_samples[::stride],
                                curve_sliding_samples[-1:],
                            ),
                            axis=0,
                        ),
                        dtype=torch.float,
                    )

                else:
                    curve_samples = torch.tensor(
                        curve_sliding_samples[::stride], dtype=torch.float
                    )
                encoder_input.append(curve_samples[:, :encoder_input_length, :])
                decoder_input.append(curve_samples[:, encoder_input_length:, 1])
                decoder_output.append(curve_samples[:, encoder_input_length:, 0])

        self.encoder_input = torch.cat(encoder_input, dim=0)
        self.decoder_input = torch.cat(decoder_input, dim=0)
        self.decoder_output = torch.cat(decoder_output, dim=0)

    def __len__(self) -> int:
        return self.encoder_input.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.encoder_input[idx],
            self.decoder_input[idx].unsqueeze(1),
            self.decoder_output[idx].unsqueeze(1),
        )
