import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from torchmetrics.regression import (
    R2Score,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)


class EncoderDecoderGRU(LightningModule):
    def __init__(
        self,
        encoder_dim: int = 2,
        decoder_dim: int = 1,
        hidden_size: int = 128,
        encoder_layers: int = 1,
        decoder_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 0.001,
        normalize: bool = True,
        voltage_min: float = 2.5,
        voltage_max: float = 4.2,
        current_min: float = -2.5,
        current_max: float = 1.0,
    ):
        super().__init__()

        self.encoder = nn.GRU(
            input_size=encoder_dim,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            dropout=dropout if encoder_layers > 1 else 0.0,
            batch_first=True,
        )

        self.decoder = nn.GRU(
            input_size=decoder_dim,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, decoder_dim),
            nn.Tanh(),  # TODO: Compare performance with and without Tanh
        )
        self.loss = nn.MSELoss()
        # Setup metrics
        self.train_r2_score, self.val_r2_score, self.test_r2_score = (
            R2Score(),
            R2Score(),
            R2Score(),
        )
        self.train_mae, self.val_mae, self.test_mae = (
            MeanAbsoluteError(),
            MeanAbsoluteError(),
            MeanAbsoluteError(),
        )
        self.train_mape, self.val_mape, self.test_mape = (
            MeanAbsolutePercentageError(),
            MeanAbsolutePercentageError(),
            MeanAbsolutePercentageError(),
        )
        self.train_rmse, self.val_rmse, self.test_rmse = (
            MeanSquaredError(squared=False),
            MeanSquaredError(squared=False),
            MeanSquaredError(squared=False),
        )
        self.decoder_layers = decoder_layers
        self.lr = lr
        self.normalize = normalize
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.current_min = current_min
        self.current_max = current_max
        self.save_hyperparameters()

    def forward(self, encoder_input, decoder_input):
        _, hidden_state = self.encoder(encoder_input)
        hidden_state = hidden_state.expand(self.decoder_layers, -1, -1).contiguous()
        output, _ = self.decoder(decoder_input, hidden_state)
        output = self.output_layer(output)
        return output

    def training_step(self, batch):
        encoder_input, decoder_input, decoder_output = batch

        predicted_output = self(encoder_input, decoder_input)
        train_loss = self.loss(predicted_output, decoder_output)
        # self.train_r2_score(predicted_output.squeeze(2), decoder_output.squeeze(2)) #TODO: Fix
        if self.normalize:
            decoder_output = (
                decoder_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
            predicted_output = (
                predicted_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
        self.train_mae(predicted_output, decoder_output)
        self.train_mape(predicted_output, decoder_output)
        self.train_rmse(predicted_output, decoder_output)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        # self.log(
        #     "train_r2_score",
        #     self.train_r2_score,
        #     on_epoch=True,
        #     on_step=True,
        #     prog_bar=True,
        # ) #TODO: Fix
        self.log(
            "train_mae", self.train_mae, on_epoch=True, on_step=True, prog_bar=True
        )
        self.log("train_mape", self.train_mape, on_epoch=True, on_step=True)
        self.log("train_rmse", self.train_rmse, on_epoch=True, on_step=True)
        return train_loss

    def validation_step(self, batch):
        encoder_input, decoder_input, decoder_output = batch

        predicted_output = self(encoder_input, decoder_input)
        val_loss = self.loss(predicted_output, decoder_output)
        # self.val_r2_score(predicted_output.squeeze(), decoder_output.squeeze()) #TODO: Fix
        if self.normalize:
            decoder_output = (
                decoder_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
            predicted_output = (
                predicted_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
        self.val_mae(predicted_output, decoder_output)
        self.val_mape(predicted_output, decoder_output)
        self.val_rmse(predicted_output, decoder_output)
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        # self.log("val_r2_score", self.val_r2_score, on_epoch=True, prog_bar=True) #TODO: Fix
        self.log("val_mae", self.val_mae, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_mape", self.val_mape, on_epoch=True, on_step=True)
        self.log("val_rmse", self.val_rmse, on_epoch=True, on_step=True)
        return val_loss

    def test_step(self, batch):
        encoder_input, decoder_input, decoder_output = batch

        predicted_output = self(encoder_input, decoder_input)
        test_loss = self.loss(predicted_output, decoder_output)
        # self.test_r2_score(predicted_output.squeeze(), decoder_output.squeeze()) #TODO: Fix
        if self.normalize:
            decoder_output = (
                decoder_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
            predicted_output = (
                predicted_output * (self.voltage_max - self.voltage_min)
                + self.voltage_min
            )
        self.test_mae(predicted_output, decoder_output)
        self.test_mape(predicted_output, decoder_output)
        self.test_rmse(predicted_output, decoder_output)
        self.log(
            "test_loss",
            test_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        # self.log("test_r2_score", self.test_r2_score, on_epoch=True, prog_bar=True) #TODO: Fix
        self.log("test_mae", self.test_mae, on_epoch=True, on_step=True, prog_bar=True)
        self.log("test_mape", self.test_mape, on_epoch=True, on_step=True)
        self.log("test_rmse", self.test_rmse, on_epoch=True, on_step=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    NUM_DIM = 2
    OUTPUT_DIM = 1
    BATCH_SIZE = 1
    m = 20
    n = 1
    encoder_input = torch.ones(BATCH_SIZE, m, NUM_DIM)
    decoder_input = torch.ones(BATCH_SIZE, n, OUTPUT_DIM)
    model = EncoderDecoderGRU()
    model.eval()
    with torch.inference_mode():
        output = model(encoder_input, decoder_input)
    assert output.shape == decoder_input.shape
