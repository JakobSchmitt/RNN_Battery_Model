from pathlib import Path
from RNN.datasets import SEFDataset
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import torch
torch.set_float32_matmul_precision('high')

class UniDataModule(LightningDataModule):
    """
    UniDataModule: A PyTorch Lightning DataModule for efficient data handling during training.
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html 

    This DataModule sets up data loaders for training, validation, and testing phases using the dataset class SEFDataset.

    Parameters:
    -----------
    mode : str
        The mode of operation, e.g., "SEFD"
    root_dir : Path
        The root directory containing the profiles' data.
    train_profiles : list, optional (default=["1", "3", "6"])
        List of profile IDs to use for training.
    val_profiles : list, optional (default=["4", "5"])
        List of profile IDs to use for validation.
    test_profiles : list, optional (default=["2"])
        List of profile IDs to use for testing.
    encoder_input_length : int, optional (default=20)
        The length of the encoder input sequence.
    stride : int, optional (default=16)
        The stride length for the sliding window.
    decoder_input_length : int, optional (default=1980)
        The length of the decoder input sequence.
    batch_size : int, optional (default=64)
        The batch size for data loaders.
    num_workers : int, optional (default=8)
        The number of worker processes to use for data loading.
    shuffle : bool, optional (default=False)
        Whether to shuffle the training data.
    normalize : bool, optional (default=True)
        Whether to normalize the voltage and current values.
    voltage_min : float, optional (default=2.5)
        The minimum voltage value for normalization.
    voltage_max : float, optional (default=4.2)
        The maximum voltage value for normalization.
    current_min : float, optional (default=-2.5)
        The minimum current value for normalization.
    current_max : float, optional (default=1.0)
        The maximum current value for normalization.
    """

    def __init__(
        self, #
        mode: str,
        root_dir: Path,
        train_profiles: list = list(map(str,[12,13])),#12, 16, 18 ] [0,2,4,6,8,10,12,14,16,18] #['2','4','6'], #['0_2.3','2_2.3','14_2.3','6_2.15','8_2.15','10_2.15'] ["0","2","5","7","8","9","10","11","12"], #["1", "3", "6"],
        val_profiles: list = list(map(str, [12,13])) , #[3,5,7,9,11,13,15,16] #['1','5'], #["1","4","6","13"], #["4", "5"],
        test_profiles: list = list(map(str, [12,13])) , #['6'], # ["3"], #["2"],
        encoder_input_length: int = 20,
        stride: int = 16,
        decoder_input_length: int = 1980,
        batch_size=64,
        num_workers=8,
        shuffle=False,
        normalize: bool = True,
        voltage_min: float = 2.5,
        voltage_max: float = 4.2,
        current_min: float = -2.5,
        current_max: float = 1.0,
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
        self.normalize = normalize
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.current_min = current_min
        self.current_max = current_max
        self.save_hyperparameters()

    def setup(self, stage=None):
        """
        Sets up the datasets for the specified stage (fit, validate, test).

        Parameters:
        -----------
        stage : str, optional
            The stage for which to set up the datasets. One of 'fit', 'validate', or 'test'.
            During training, "fit" is passed as the stage argument
        """
        if stage == "fit":
            if self.mode == "SEFD":
                self.train_dataset = SEFDataset(
                    root_dir=self.root_dir,
                    profiles=self.train_profiles,
                    encoder_input_length=self.encoder_input_length,
                    decoder_input_length=self.decoder_input_length,
                    stride=self.stride,
                    normalize=self.normalize,
                    voltage_min=self.voltage_min,
                    voltage_max=self.voltage_max,
                    current_min=self.current_min,
                    current_max=self.current_max,
                )
                self.val_dataset = SEFDataset(
                    root_dir=self.root_dir,
                    profiles=self.val_profiles,
                    encoder_input_length=self.encoder_input_length,
                    decoder_input_length=self.decoder_input_length,
                    stride=self.stride,
                    normalize=self.normalize,
                    voltage_min=self.voltage_min,
                    voltage_max=self.voltage_max,
                    current_min=self.current_min,
                    current_max=self.current_max,
                )
        elif stage == "validate":
            if self.mode == "SEFD":
                self.val_dataset = SEFDataset(
                    root_dir=self.root_dir,
                    profiles=self.val_profiles,
                    encoder_input_length=self.encoder_input_length,
                    decoder_input_length=self.decoder_input_length,
                    stride=self.stride,
                    normalize=self.normalize,
                    voltage_min=self.voltage_min,
                    voltage_max=self.voltage_max,
                    current_min=self.current_min,
                    current_max=self.current_max,
                )
        elif stage == "test":
            if self.mode == "SEFD":
                self.test_dataset = SEFDataset(
                    root_dir=self.root_dir,
                    profiles=self.test_profiles,
                    encoder_input_length=self.encoder_input_length,
                    decoder_input_length=self.decoder_input_length,
                    stride=self.stride,
                    normalize=self.normalize,
                    voltage_min=self.voltage_min,
                    voltage_max=self.voltage_max,
                    current_min=self.current_min,
                    current_max=self.current_max,
                )
        else:
            raise ValueError(f"Data Module not designed for stage : {stage}")

    def train_dataloader(self):
        """Returns a DataLoader for the training dataset."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        """Returns a DataLoader for the validation dataset."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True

        )

    def test_dataloader(self):
        """Returns a DataLoader for the test dataset."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True

        )