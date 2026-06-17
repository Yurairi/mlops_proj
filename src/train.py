import argparse
from pathlib import Path

import joblib
import pandas as pd

from clearml import Dataset, OutputModel, Task
from clearml import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_split(base_path: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(Path(base_path) / filename)


def infer_pos_label(series):
    vals = set(series.unique())
    if "positive" in vals:
        return "positive"
    if 1 in vals:
        return 1
    return sorted(list(vals))[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="val.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--task_name", type=str, default="experiment-1")
    args = parser.parse_args()

    task = Task.init(
        project_name="sentiment-lab",
        task_name=args.task_name,
        output_uri=True,
    )
    task.execute_remotely(queue_name="students")
    task.connect(vars(args))

    logger = task.get_logger()
    output_model = OutputModel(task=task, framework="scikit-learn")

    dataset = Dataset.get(dataset_id=args.dataset_id)
    base_path = dataset.get_local_copy()

    train_df = load_split(base_path, args.train_csv)
    val_df = load_split(base_path, args.val_csv)
    test_df = load_split(base_path, args.test_csv)

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_val = val_df["text"]
    y_val = val_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    vec = TfidfVectorizer(max_features=args.max_features)
    X_train_vec = vec.fit_transform(X_train)
    X_val_vec = vec.transform(X_val)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter)
    model.fit(X_train_vec, y_train)

    val_pred = model.predict(X_val_vec)
    test_pred = model.predict(X_test_vec)

    pos_label = infer_pos_label(y_train)

    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, pos_label=pos_label)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, pos_label=pos_label)

    logger.report_scalar("metrics", "val_accuracy", value=val_acc, iteration=0)
    logger.report_scalar("metrics", "val_f1", value=val_f1, iteration=0)
    logger.report_scalar("metrics", "test_accuracy", value=test_acc, iteration=0)
    logger.report_scalar("metrics", "test_f1", value=test_f1, iteration=0)

    labels = sorted(list(set(y_train) | set(y_val) | set(y_test)))
    cm = confusion_matrix(y_test, test_pred, labels=labels)

    logger.report_confusion_matrix(
        title="confusion_matrix",
        series="test",
        iteration=0,
        matrix=cm,
        xaxis="Predicted",
        yaxis="True",
        yaxis_reversed=True,
    )

    model_path = "model.pkl"
    joblib.dump({"vectorizer": vec, "model": model}, model_path)
    output_model.update_weights(weights_filename=model_path)

    if isinstance(pos_label, str):
        output_model.update_labels({str(lbl): i for i, lbl in enumerate(labels)})

    print(f"val_accuracy={val_acc:.4f}, val_f1={val_f1:.4f}")
    print(f"test_accuracy={test_acc:.4f}, test_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
