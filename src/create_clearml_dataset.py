from pathlib import Path

from clearml import Dataset

DATA_DIR = Path("data")
PROJECT = "MLOps_Course"
NAME = "IMDb_Sentiment_Classification"
VERSION = "1.0.0"


def create_clearml_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError("data/ directory not found")

    ds = Dataset.create(
        dataset_name=NAME,
        dataset_project=PROJECT,
        dataset_version=VERSION,
        description="IMDb movie reviews dataset for sentiment classification",
    )

    ds.add_files(path=str(DATA_DIR / "train.csv"))
    ds.add_files(path=str(DATA_DIR / "val.csv"))
    ds.add_files(path=str(DATA_DIR / "test.csv"))
    ds.add_files(path=str(DATA_DIR / "dataset_metadata.json"))

    ds.upload()
    ds.finalize()
    return ds.id


if __name__ == "__main__":
    dataset_id = create_clearml_dataset()
    print(dataset_id)
