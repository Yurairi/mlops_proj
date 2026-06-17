import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import os
import json


def prepare_imdb_dataset():
    """
    Скачивает и подготавливает датасет IMDb для классификации текстов
    """
    print("Загрузка IMDb датасета...")

    dataset = load_dataset("imdb")

    train_data = dataset["train"]
    test_data = dataset["test"]

    train_df = pd.DataFrame({"text": train_data["text"], "label": train_data["label"]})

    test_df = pd.DataFrame({"text": test_data["text"], "label": test_data["label"]})

    train_sample = train_df.sample(n=5000, random_state=42)
    test_sample = test_df.sample(n=1000, random_state=42)

    train_data, val_data = train_test_split(
        train_sample, test_size=0.2, random_state=42, stratify=train_sample["label"]
    )

    os.makedirs("data", exist_ok=True)

    train_data.to_csv("data/train.csv", index=False)
    val_data.to_csv("data/val.csv", index=False)
    test_sample.to_csv("data/test.csv", index=False)

    print(f"Датасет подготовлен:")
    print(f"   - Train: {len(train_data)} samples")
    print(f"   - Validation: {len(val_data)} samples")
    print(f"   - Test: {len(test_sample)} samples")
    print(f"   - Сохранен в папке 'data/'")

    metadata = {
        "dataset_name": "IMDb Sentiment",
        "num_classes": 2,
        "class_names": ["negative", "positive"],
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_sample),
        "description": "IMDb movie reviews for sentiment classification",
    }

    with open("data/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return train_data, val_data, test_sample


if __name__ == "__main__":
    prepare_imdb_dataset()
    print("\nГотово!")
