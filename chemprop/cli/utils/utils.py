from typing import Any, List, Union
import csv, tqdm, logging, os, pickle
import numpy as np
from chemprop.data import MoleculeDataset

__all__ = ["pop_attr"]


def pop_attr(o: object, attr: str, *args) -> Any | None:
    """like ``pop()`` but for attribute maps"""
    match len(args):
        case 0:
            return _pop_attr(o, attr)
        case 1:
            return _pop_attr_d(o, attr, args[0])
        case _:
            raise TypeError(f"Expected at most 2 arguments! got: {len(args)}")


def _pop_attr(o: object, attr: str) -> Any:
    val = getattr(o, attr)
    delattr(o, attr)

    return val


def _pop_attr_d(o: object, attr: str, default: Any | None = None) -> Any | None:
    try:
        val = getattr(o, attr)
        delattr(o, attr)
    except AttributeError:
        val = default

    return val


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def preprocess_smiles_columns(path: str,
                              smiles_columns: Union[str, List[str]] = None,
                              number_of_molecules: int = 1) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if isinstance(smiles_columns, str):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns


def save_smiles_splits(
    data_path: str,
    save_dir: str,
    task_names: List[str] = None,
    features_path: List[str] = None,
    constraints_path: str = None,
    train_data: MoleculeDataset = None,
    val_data: MoleculeDataset = None,
    test_data: MoleculeDataset = None,
    smiles_columns: List[str] = None,
    loss_function: str = None,
    logger: logging.Logger = None,
) -> None:
    """
    Saves a csv file with train/val/test splits of target data and additional features.
    Also saves indices of train/val/test split as a pickle file. Pickle file does not support repeated entries
    with the same SMILES or entries entered from a path other than the main data path, such as a separate test path.

    :param data_path: Path to data CSV file.
    :param save_dir: Path where pickle files will be saved.
    :param task_names: List of target names for the model as from the function get_task_names().
        If not provided, will use datafile header entries.
    :param features_path: List of path(s) to files with additional molecule features.
    :param constraints_path: Path to constraints applied to atomic/bond properties prediction.
    :param train_data: Train :class:`~chemprop.data.data.MoleculeDataset`.
    :param val_data: Validation :class:`~chemprop.data.data.MoleculeDataset`.
    :param test_data: Test :class:`~chemprop.data.data.MoleculeDataset`.
    :param smiles_columns: The name of the column containing SMILES. By default, uses the first column.
    :param loss_function: The loss function to be used in training.
    :param logger: A logger for recording output.
    """
    makedirs(save_dir)

    info = logger.info if logger is not None else print
    save_split_indices = True

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=data_path, smiles_columns=smiles_columns)

    with open(data_path) as f:
        f = open(data_path)
        reader = csv.DictReader(f)

        indices_by_smiles = {}
        for i, row in enumerate(tqdm(reader)):
            smiles = tuple([row[column] for column in smiles_columns])
            if smiles in indices_by_smiles:
                save_split_indices = False
                info(
                    "Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated."
                )
                break
            indices_by_smiles[smiles] = i

    if task_names is None:
        task_names = get_task_names(
            path=data_path,
            smiles_columns=smiles_columns,
            loss_function=loss_function,
            )

    if loss_function == "quantile_interval":
        num_tasks = len(task_names) // 2
        task_names = task_names[:num_tasks]

    features_header = []
    if features_path is not None:
        extension_sets = set([os.path.splitext(feat_path)[1] for feat_path in features_path])
        if extension_sets == {'.csv'}:
            for feat_path in features_path:
                with open(feat_path, "r") as f:
                    reader = csv.reader(f)
                    feat_header = next(reader)
                    features_header.extend(feat_header)

    if constraints_path is not None:
        with open(constraints_path, "r") as f:
            reader = csv.reader(f)
            constraints_header = next(reader)

    all_split_indices = []
    for dataset, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f"{name}_smiles.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            if smiles_columns[0] == "":
                writer.writerow(["smiles"])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f"{name}_full.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                targets = [x.tolist() if isinstance(x, np.ndarray) else x for x in dataset_targets[i]]
                # correct the number of targets when running quantile regression
                targets = targets[:len(task_names)]
                writer.writerow(smiles + targets)

        if features_path is not None:
            dataset_features = dataset.features()
            if extension_sets == {'.csv'}:
                with open(os.path.join(save_dir, f"{name}_features.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(features_header)
                    writer.writerows(dataset_features)
            else:
                np.save(os.path.join(save_dir, f"{name}_features.npy"), dataset_features)

        if constraints_path is not None:
            dataset_constraints = [d.raw_constraints for d in dataset._data]
            with open(os.path.join(save_dir, f"{name}_constraints.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(constraints_header)
                writer.writerows(dataset_constraints)

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(
                        f"Warning: SMILES string in {name} could not be found in data file, and "
                        "likely came from a secondary data file. The pickle file of split indices "
                        "can only indicate indices for a single file and will not be generated."
                    )
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

        if name == "train":
            data_weights = dataset.data_weights()
            if any([w != 1 for w in data_weights]):
                with open(os.path.join(save_dir, f"{name}_weights.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["data weights"])
                    for weight in data_weights:
                        writer.writerow([weight])

    if save_split_indices:
        with open(os.path.join(save_dir, "split_indices.pckl"), "wb") as f:
            pickle.dump(all_split_indices, f)