from pathlib import Path
from RNN.datasets import SEFDataset
from torch.utils.data import DataLoader
from lightning import LightningDataModule


# TODO: Pass normalization arguments instead of assuming defaults
class UniDataModule(LightningDataModule):
    def __init__(
        self,
        mode: str,
        root_dir: Path,
        train_profiles: list = ["1", "3", "6"],
        val_profiles: list = ["4", "5"],
        test_profiles: list = ["2"],
        encoder_input_length: int = 20,
        stride: int = 35,
        decoder_input_length: int = 1_980,
        batch_size=64,
        num_workers=8,
        shuffle=False,
    ):
        super().__init__()
        self.mode = mode
        self.root_dir = root_dir
        self.train_profiles = train_profiles
        self.val_profiles = val_profiles
        self.test_profiles = test_profiles
        self.encoder_input_length = encoder_input_length
        self.stride = stride
        self.decoder_input_length = decoder_input_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            if self.mode == "SEFD":
                self.train_dataset = SEFDataset(
                    root_dir=self.root_dir,
                    profiles=self.train_profiles,
                    encoder_input_length=self.encoder_input_length,
                    decoder_input_length=self.decoder_input_length,
                    stride=self.stride,
                )
                self.val_dataset = SEFDataset(
                    root_dir=self.root_dir,
                    profiles=self.val_profiles,
                    encoder_input_length=self.encoder_input_length,
                    decoder_input_length=self.decoder_input_length,
                    stride=self.stride,
                )
        elif stage == "validate":
            if self.mode == "SEFD":
                self.val_dataset = SEFDataset(
                    root_dir=self.root_dir,
                    profiles=self.val_profiles,
                    encoder_input_length=self.encoder_input_length,
                    decoder_input_length=self.decoder_input_length,
                    stride=self.stride,
                )
        elif stage == "predict":
            raise ValueError("Data Module not designed for stage='predict'")
        elif stage == "test":
            if self.mode == "SEFD":
                self.test_dataset = SEFDataset(
                    root_dir=self.root_dir,
                    profiles=self.test_profiles,
                    encoder_input_length=self.encoder_input_length,
                    decoder_input_length=self.decoder_input_length,
                    stride=self.stride,
                )
        else:
            raise ValueError(f"Data Module not designed for stage : {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    data_module = UniDataModule(
        mode="SEFD",
        root_dir=Path(
            "/home/mazin/Projects/Thesis/Data/battery_model_paper/preprocessed_small_20/149"
        ),
        train_profiles=["1", "3", "6"],
        val_profiles=["4", "5"],
        test_profiles=["2"],
        encoder_input_length=20,
        stride=35,
        decoder_input_length=1980,
        batch_size=8,
        num_workers=8,
        shuffle=False,
    )
    data_module.setup("fit")
