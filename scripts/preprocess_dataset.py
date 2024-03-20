import hydra
import logging
import pandas as pd
from pathlib import Path


@hydra.main(
    version_base="1.1", config_path="../config/", config_name="preprocess_dataset"
)
def main(cfg):
    cfg.data_dir = Path(hydra.utils.to_absolute_path(cfg.data_dir))

    for profile in cfg.profiles:
        curves_dir = sorted((cfg.data_dir / cfg.cell_id / profile).iterdir())
        for curve_dir in curves_dir:
            if curve_dir.stem not in [f"{profile}_4_0"]:
                continue
            df = pd.read_csv(curve_dir)
            logging.info(f"Directory : {curve_dir}")
            logging.info(f"Shape before resampling : {df.shape[0]}")
            df["Time[s]"] = pd.to_datetime(df["Time[s]"], unit="s")
            df = (
                df.resample(cfg.resampling_rate, on="Time[s]")
                .mean()
                .interpolate(method="linear")
            )
            logging.info(f"Shape after resampling : {df.shape[0]}")

            logging.info(
                f"Max length after splitting : {df.shape[0] / cfg.num_sequences}"
            )

            for i in range(cfg.num_sequences):
                save_dir = cfg.data_dir / Path(cfg.save_folder) / cfg.cell_id / profile
                save_dir.mkdir(parents=True, exist_ok=True)
                df.iloc[i :: cfg.num_sequences].to_csv(
                    save_dir / Path(curve_dir.stem + f"_{i}.csv")
                )


if __name__ == "__main__":
    main()
