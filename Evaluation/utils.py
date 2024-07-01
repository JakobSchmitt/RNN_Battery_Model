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

def plot_prediction(
    encoder_current_list: List[torch.Tensor],
    decoder_current_list: List[torch.Tensor],
    encoder_voltage_list: List[torch.Tensor],
    predicted_voltage_list: List[torch.Tensor],
    title: str,
    show_legends: bool = False,  # New argument to control legends visibility
):
    num_plots = len(encoder_current_list)  # Number of plots
    cmap = get_cmap("tab20")  # Choose a color map, e.g., 'tab10'

    # Create subplots
    fig, ax = plt.subplots(2, figsize=(24, 12), sharex=True)
    fig.suptitle(title)
    for i in range(num_plots):
        encoder_current = encoder_current_list[i]
        decoder_current = decoder_current_list[i]
        encoder_voltage = encoder_voltage_list[i]
        predicted_voltage = predicted_voltage_list[i]

        encoder_input_length = len(encoder_current)
        decoder_input_length = len(decoder_current)

        # Calculate time in minutes
        time_per_sample = 1 / 0.5 / 60  # 0.5Hz -> seconds -> minutes
        encoder_time = np.arange(encoder_input_length) * time_per_sample
        decoder_time = (
            np.arange(decoder_input_length) * time_per_sample + encoder_time[-1]
        )

        color = cmap(i)  # Get color from the color map

        # Plot the first set of data on the first subplot
        ax[0].plot(
            encoder_time, encoder_current, label=f"Encoder Current_{i}", color=color
        )
        ax[0].plot(
            decoder_time, decoder_current, label=f"Decoder Current_{i}", color=color
        )
        ax[0].set_ylim(-3.5, 3.5)
        ax[0].set_ylabel("Current")
        if show_legends:  # Only add legend if show_legends is True
            ax[0].legend()
        ax[0].grid(True)

        # Plot the second set of data on the second subplot
        ax[1].plot(
            encoder_time, encoder_voltage, label=f"Encoder Voltage_{i}", color=color
        )
        ax[1].plot(
            decoder_time, predicted_voltage, label=f"Predicted Voltage_{i}", color=color
        )
        ax[1].set_ylim(2, 4.5)
        ax[1].set_xlabel("Time (minutes)")
        ax[1].set_ylabel("Voltage")
        if show_legends:  # Only add legend if show_legends is True
            ax[1].legend()
        ax[1].grid(True)

    # Add a dashed line between the two subplots
    line_position = encoder_time[-1]
    ax[0].axvline(line_position, color="red", linestyle="--")
    ax[1].axvline(line_position, color="red", linestyle="--")


def behavior_prediction(
    rnn_model,
    encoder_current_value: float,
    encoder_voltage_value: float,
    decoder_current_value: float,
    encoder_input_length: int = 20,
    decoder_input_length: int = 1980,
    current_min: float = -2.5,
    current_max: float = 1,
    voltage_min: float = 2.5,
    voltage_max: float = 4.2,
):
    encoder_current = torch.full(
        size=(1, encoder_input_length, 1),
        fill_value=encoder_current_value,
        dtype=torch.float,
    )
    encoder_current_normalized = normalize(encoder_current, current_min, current_max)
    encoder_voltage = torch.full(
        size=(1, encoder_input_length, 1),
        fill_value=encoder_voltage_value,
        dtype=torch.float,
    )
    encoder_voltage_normalized = normalize(encoder_voltage, voltage_min, voltage_max)
    encoder_input_normalized = torch.cat(
        [encoder_voltage_normalized, encoder_current_normalized], dim=2
    )

    decoder_current = torch.full(
        size=(1, decoder_input_length, 1),
        fill_value=decoder_current_value,
        dtype=torch.float,
    )
    decoder_current_normalized = normalize(decoder_current, current_min, current_max)
    predicted_voltage_normalized = rnn_model(
        encoder_input_normalized, decoder_current_normalized
    )
    predicted_voltage = denormalize(
        predicted_voltage_normalized, voltage_min, voltage_max
    )

    return (
        encoder_current.detach().numpy().flatten(),
        decoder_current.detach().numpy().flatten(),
        encoder_voltage.detach().numpy().flatten(),
        predicted_voltage.detach().numpy().flatten(),
    )


def normalize(tensor, min_value=2.5, max_value=4.2):
    return (tensor - min_value) / (max_value - min_value)


def denormalize(tensor, min_value=2.5, max_value=4.2):
    return tensor * (max_value - min_value) + min_value


def perform_multiple_predictions(
    model: EncoderDecoderGRU,
    encoder_current_values: list,
    encoder_voltage_values: list,
    decoder_current_values: list,
    encoder_input_length: int = 20,
    decoder_input_length: int = 1980,
    current_min: float = -2.5,
    current_max: float = 1.0,
    voltage_min: float = 2.5,
    voltage_max: float = 4.2,
) -> tuple[list, list, list, list]:
    encoder_current_list = []
    decoder_current_list = []
    encoder_voltage_list = []
    predicted_voltage_list = []

    for encoder_current, encoder_voltage, decoder_current in zip(
        encoder_current_values, encoder_voltage_values, decoder_current_values
    ):
        (
            encoder_current_result,
            decoder_current_result,
            encoder_voltage_result,
            predicted_voltage_result,
        ) = behavior_prediction(
            model,
            encoder_current,
            encoder_voltage,
            decoder_current,
            encoder_input_length=encoder_input_length,
            decoder_input_length=decoder_input_length,
            current_min=current_min,
            current_max=current_max,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
        )

        encoder_current_list.append(encoder_current_result)
        decoder_current_list.append(decoder_current_result)
        encoder_voltage_list.append(encoder_voltage_result)
        predicted_voltage_list.append(predicted_voltage_result)

    return (
        encoder_current_list,
        decoder_current_list,
        encoder_voltage_list,
        predicted_voltage_list,
    )
