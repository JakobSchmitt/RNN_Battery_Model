import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule


class EncoderDecoderGRU(LightningModule):
    def __init__(
        self,
        encoder_input_dim: int = 2,
        output_dim: int = 1,
        hidden_size: int = 128,
        encoder_layers: int = 1,
        decoder_layers: int = 2,
        dropout: float = 0.3,
        batch_first: bool = False,
    ):
        super().__init__()

        self.encoder = nn.GRU(
            input_size=encoder_input_dim,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            dropout=dropout if encoder_layers > 1 else 0.0,
            batch_first=batch_first,
        )

        self.decoder = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.decoder_layers = decoder_layers

        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_dim), nn.Tanh())

    def forward(self, encoder_input, decoder_input):
        _, hidden_state = self.encoder(encoder_input)
        hidden_state = hidden_state.expand(self.decoder_layers, -1, -1)
        output, _ = self.decoder(decoder_input, hidden_state)
        output = self.output_layer(output)

        return output


if __name__ == "__main__":
    torch.manual_seed(42)
    NUM_DIM = 2
    OUTPUT_DIM = 1
    BATCH_SIZE = 1
    HIDDEN_SIZE = 128
    m = 20
    n = 21
    model = EncoderDecoderGRU()
    encoder_input = torch.ones(m, BATCH_SIZE, NUM_DIM)
    decoder_input = torch.ones(n, BATCH_SIZE, OUTPUT_DIM)
    model.eval()
    with torch.inference_mode():
        output = model(encoder_input, decoder_input)
    assert output.shape == decoder_input.shape
    print(output)
