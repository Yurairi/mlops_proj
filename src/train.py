import argparse
from pathlib import Path

import joblib
import pandas as pd
from clearml import Dataset, Logger, OutputModel, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


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

    Task.add_requirements("numpy", "2.4.6")
    Task.add_requirements("scikit-learn", "1.9.0")
    Task.add_requirements("pandas", "3.0.3")
    Task.add_requirements("joblib", "1.3.2")

    task = Task.init(
        project_name="Text Classification",
        task_name=args.task_name,
        output_uri=True,
        auto_connect_arg_parser=parser,
    )
    task.connect(vars(args))

    logger: Logger = task.get_logger()
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

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=args.max_features)),
            (
                "clf",
                LogisticRegression(
                    C=args.C,
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    val_pred = pipe.predict(X_val)
    test_pred = pipe.predict(X_test)

    pos_label = infer_pos_label(y_train)

    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, pos_label=pos_label)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, pos_label=pos_label)

    logger.report_scalar("metrics", "accuracy", value=test_acc, iteration=0)
    logger.report_scalar("metrics", "f1", value=test_f1, iteration=0)
    logger.report_scalar("metrics", "val_accuracy", value=val_acc, iteration=0)
    logger.report_scalar("metrics", "val_f1", value=val_f1, iteration=0)

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
    joblib.dump(pipe, model_path, compress=True)
    output_model.update_weights(weights_filename=model_path)

    print(f"val_accuracy={val_acc:.4f}, val_f1={val_f1:.4f}")
    print(f"test_accuracy={test_acc:.4f}, test_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
