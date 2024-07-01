import torch
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from RNN.models import EncoderDecoderGRU
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from numpy.lib.stride_tricks import sliding_window_view


# TODO: Fails on GPU / only works for RNN Model
def profile_predict(
    model: EncoderDecoderGRU,
    profile_path: str,
    encoder_input_length=20,
    decoder_input_length=1980,
    voltage_min=2.5,
    voltage_max=4.2,
    current_min=-2.5,
    current_max=1,
):
    curve = pd.read_csv(profile_path, usecols=["voltage", "current"])
    last_sample_length = curve[encoder_input_length:].shape[0] % decoder_input_length
    # Apply MinMax Normalization -> [0,1]
    curve["voltage"] = (curve["voltage"] - voltage_min) / (voltage_max - voltage_min)
    curve["current"] = (curve["current"] - current_min) / (current_max - current_min)
    actual_current = torch.tensor(curve["current"], dtype=torch.float)
    actual_voltage = torch.tensor(curve["voltage"], dtype=torch.float)
    predicted_voltage = torch.zeros(curve.shape[0], dtype=torch.float)
    curve_sliding_samples = sliding_window_view(
        curve[encoder_input_length:], (decoder_input_length, 2)
    ).squeeze()
    curve_samples = torch.tensor(
        np.concatenate(
            (
                curve_sliding_samples[::decoder_input_length],
                curve_sliding_samples[-1:],
            ),
            axis=0,
        ),
        dtype=torch.float,
    )
    encoder_input = torch.tensor(
        curve[:encoder_input_length].to_numpy(), dtype=torch.float
    ).unsqueeze(0)

    decoder_input = curve_samples[0, :, 1].reshape(1, -1, 1)
    predicted_voltage[:encoder_input_length] = encoder_input[:, :, 0]
    model.eval()
    with torch.inference_mode():
        output = model(encoder_input, decoder_input)
    predicted_voltage[
        encoder_input_length : encoder_input_length + decoder_input_length
    ] = output.squeeze()
    for i in range(1, curve_samples.shape[0]):
        if i != curve_samples.shape[0] - 1:
            encoder_input = torch.cat(
                [
                    output[:, -encoder_input_length:],
                    decoder_input[:, -encoder_input_length:],
                ],
                dim=-1,
            )
            decoder_input = curve_samples[i, :, 1].reshape(1, -1, 1)
            model.eval()
            with torch.inference_mode():
                output = model(encoder_input, decoder_input)
            predicted_voltage[
                encoder_input_length
                + i * decoder_input_length : encoder_input_length
                + (i + 1) * decoder_input_length
            ] = output.squeeze()
        else:
            encoder_input = torch.cat(
                [
                    output[
                        :,
                        last_sample_length
                        - decoder_input_length
                        - encoder_input_length : last_sample_length
                        - decoder_input_length,
                    ],
                    decoder_input[
                        :,
                        last_sample_length
                        - decoder_input_length
                        - encoder_input_length : last_sample_length
                        - decoder_input_length,
                    ],
                ],
                dim=-1,
            )
            decoder_input = curve_samples[-1, :, 1].reshape(1, -1, 1)
            model.eval()
            with torch.inference_mode():
                output = model(encoder_input, decoder_input)
            predicted_voltage[-decoder_input_length:] = output.squeeze()
            # Denormalize after prediction
            actual_current = actual_current * (current_max - current_min) + current_min
            predicted_voltage = (
                predicted_voltage * (voltage_max - voltage_min) + voltage_min
            )
            actual_voltage = actual_voltage * (voltage_max - voltage_min) + voltage_min

    return actual_current, actual_voltage, predicted_voltage


def plot_profile(
    actual_current,
    actual_voltage,
    predicted_voltage,
    title: str = "Actual vs Predicted Voltage",
    sampling_rate: float = 0.5,  # Sampling rate in Hz
):
    # Create time axis in minutes
    time_axis = np.arange(len(actual_current)) / (sampling_rate * 60)

    fig, ax = plt.subplots(2, figsize=(20, 8))
    fig.suptitle(title)

    # Plot actual and predicted voltages
    ax[0].plot(time_axis, actual_current, label="Actual Current")
    ax[1].plot(time_axis, actual_voltage, label="Actual Voltage")
    ax[1].plot(time_axis, predicted_voltage, label="Predicted Voltage")

    # Calculate R^2 score and MAE
    r2 = r2_score(actual_voltage, predicted_voltage)
    mae = mean_absolute_error(actual_voltage, predicted_voltage)

    # Add R^2 score and MAE to legend
    ax[1].legend(
        [
            "Actual Voltage",
            f"Predicted Voltage (R2 : {r2:.3f}, MAE : {mae:.3f})",
        ]
    )

    # Set axis label
    ax[1].set_xlabel("Time (minutes)")
    ax[0].set_ylabel("Current")
    ax[1].set_ylabel("Voltage")

    # Show plot
    plt.show()


def calculate_metrics(actual_voltage: list, predicted_voltage: list):
    # Concatenate lists if they have multiple elements
    actual_voltage = np.concatenate(actual_voltage)
    predicted_voltage = np.concatenate(predicted_voltage)

    r2 = r2_score(actual_voltage, predicted_voltage)
    rmse = root_mean_squared_error(actual_voltage, predicted_voltage)
    mae = mean_absolute_error(actual_voltage, predicted_voltage)
    mape = mean_absolute_percentage_error(actual_voltage, predicted_voltage)
    print("R-squared score:", r2)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape)