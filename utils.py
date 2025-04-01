from datasets import DatasetDict
from datasets import Dataset

def split_dataset(
    ds: Dataset, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42
) -> DatasetDict:
    """
    Splits the dataset into train, validation, and test sets.

    Parameters:
        ds: The input dataset to be split (should be a Dataset object).
        val_size: Proportion of the dataset to include in the validation split.
        test_size: Proportion of the dataset to include in the test split.
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict: A dictionary containing the train, validation, and test splits.
    """
    eval_size = val_size + test_size
    eval_prop = test_size / (test_size + val_size)
    train_eval = ds.train_test_split(test_size=eval_size, seed=seed)
    val_test = train_eval["test"].train_test_split(test_size=eval_prop, seed=seed)
    ds_splits = DatasetDict(
        {
            "train": train_eval["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    return ds_splits

def normalize(ds: Dataset, col: str) -> Dataset:
    """
    Min-max normalize a column in a Hugging Face dataset to the [0, 1] range.

    Args:
        dataset (Dataset): The Hugging Face dataset to normalize.
        column (str): Name of the column to normalize.

    Returns:
        Dataset: The dataset with the new normalized column as "{column}_norm".
    """
    new_column = f"{col}_norm"

    col_values = ds[col]
    min_val = min(col_values)
    max_val = max(col_values)

    # Handle constant columns
    if min_val == max_val:
        return ds.map(lambda x: {new_column: 0.5})

    # Normalization function
    def normalize(data):
        data[new_column] = (data[col] - min_val) / (max_val - min_val)
        return data

    return ds.map(normalize)