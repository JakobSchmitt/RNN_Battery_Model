import torch
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


class SEFDataset(Dataset):
    """
    SlidingEncoder - Uses stride to generate encoder input position in the time series
    FixedDecoder - After fixing the encoder input position, generate decoder input of fixed length
    """

    def __init__(
        self,
        root_dir: Path,
        profiles: List[int],
        encoder_input_length: int = 20,
        decoder_input_length: int = 2000,
        stride: int = 35,
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
                    # Apply MinMax Normalization -> [0,1]
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


if __name__ == "__main__":
    dataset = SEFDataset(
        root_dir=Path(
            "/home/mazin/Projects/Thesis/Data/battery_model_paper/preprocessed_small_20/149"
        ),
        profiles=[2],
    )
