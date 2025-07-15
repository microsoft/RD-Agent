from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):

    # Create train and test splits from train set
    old_train = pd.read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Create sample submission
    sample_submission = new_test.copy()
    sample_submission["price"] = 43878.016
    sample_submission.drop(sample_submission.columns.difference(["id", "price"]), axis=1, inplace=True)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Create private files
    new_test.to_csv(private / "submission_test.csv", index=False)

    # Create public files visible to agents
    new_train.to_csv(public / "train.csv", index=False)
    new_test.drop(["price"], axis=1, inplace=True)
    new_test.to_csv(public / "test.csv", index=False)

    # Checks
    assert new_test.shape[1] == 12, "Public test set should have 12 columns"
    assert new_train.shape[1] == 13, "Public train set should have 13 columns"
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Length of new_train and new_test should equal length of old_train"


if __name__ == "__main__":
    competitions = "playground-series-s4e9"
    raw = Path(__file__).resolve().parent
    prepare(
        raw=raw,
        public=raw.parent.parent / competitions,
        private=raw.parent.parent / "eval" / competitions,
    )
