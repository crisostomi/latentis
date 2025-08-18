from typing import Optional, Sequence

import datasets
from datasets import ClassLabel, DatasetDict, Dataset
from datasets import Features, Value, Image

from latentis.data.dataset import DatasetView, Feature, FeatureMapping, HFDatasetView
from latentis.data.imagenet import read_imagenet_labels
import os
import glob

class LoadLocalDataset:
    def __init__(self, loader, path):
        self.loader = loader
        self.path = path

    def __call__(self, **kwargs):
        if not callable(self.loader):
            raise ValueError("loader must be a callable returning a Dataset or DatasetDict")
        return self.loader(self.path)

class LoadHFDataset:
    def __init__(self, **load_params) -> None:
        self.load_params = load_params

    def __call__(self) -> DatasetDict:
        path = self.load_params.get("path")
        name = self.load_params.get("name")

        if path == "BeIR/nq" and name == "corpus":
                corpus = datasets.load_dataset(path=path, name=name)["corpus"]
                corpus_dict = corpus.train_test_split(test_size=65536, seed=42)
                train = corpus_dict["train"]
                val   = corpus_dict["test"]
                train = train.shuffle(seed=10).select(range(2_000_000))

                return DatasetDict({
                    "train": train,
                    "test":  val
                })

        return datasets.load_dataset(**self.load_params)


def load_hf_dataset(**load_params) -> DatasetDict:
    return datasets.load_dataset(**load_params)


def subset(
    data: DatasetDict, perc: float, seed: int, stratify_by_column: str = None, **kwargs
) -> DatasetDict:
    """Subset a dataset."""
    if perc == 1:
        return data

    return DatasetDict(
        {
            split: data[split].train_test_split(
                test_size=perc,
                seed=seed,
                stratify_by_column=stratify_by_column,
                **kwargs,
            )["test"]
            for split in data.keys()
        }
    )


class Subset:
    def __init__(self, perc: float, seed: int, **kwargs) -> None:
        self.perc = perc
        self.seed = seed
        self.kwargs = kwargs

    def __call__(self, data: DatasetDict) -> DatasetDict:
        return subset(data, perc=self.perc, seed=self.seed, **self.kwargs)


def select_columns(data: DatasetDict, columns: Sequence[str]) -> DatasetDict:
    """Select columns from a dataset."""
    to_prune = {col for cols in data.column_names.values() for col in cols} - set(
        columns
    )
    return data.remove_columns(to_prune)


def store_to_disk(data: DatasetDict, path: str) -> DatasetDict:
    """Store a dataset to disk."""
    data.save_to_disk(path)
    return data


def add_id_column(data: DatasetDict, id_column: str) -> DatasetDict:
    """Add an id column to a dataset."""
    return data.map(
        lambda _, index: {id_column: index},
        with_indices=True,
    )


class HFMap:
    def __init__(self, input_columns: Sequence[str], fn):
        self.input_columns = input_columns
        self.fn = fn

    def __call__(self, data: DatasetDict) -> DatasetDict:
        data = data.map(
            self.fn,
            input_columns=self.input_columns,
            batched=True,
        )
        return data


class PruneColumns:
    def __init__(
        self,
        keep: Optional[Sequence[str]] = None,
        remove: Optional[Sequence[str]] = None,
    ) -> None:
        if keep is not None and remove is not None:
            raise ValueError("Cannot specify both `keep` and `remove`.")
        if keep is None and remove is None:
            raise ValueError("Must specify either `keep` or `remove`.")

        self.keep = keep
        self.remove = remove

    def __call__(self, data: DatasetDict) -> DatasetDict:
        if self.keep is not None:
            to_keep = self.keep
        else:
            to_keep = set(data.column_names) - set(self.remove)

        return select_columns(data, columns=to_keep)


class MapFeatures:
    def __init__(self, *feature_mappings: FeatureMapping) -> None:
        self.feature_mappings = feature_mappings

    def __call__(self, data: DatasetDict) -> DatasetDict:
        data = data.map(
            lambda *source_col_vals: {
                target_col: source_col_val
                for source_col_val, target_col in zip(
                    source_col_vals,
                    [
                        feature_mapping.target_col
                        for feature_mapping in self.feature_mappings
                    ],
                )
            },
            batched=True,
            input_columns=[
                feature_mapping.source_col for feature_mapping in self.feature_mappings
            ],
        )
        return data


class FeatureCast:
    def __init__(self, feature_name: str, feature) -> None:
        self.feature_name = feature_name
        self.feature = feature

    def __call__(self, data: DatasetDict) -> DatasetDict:
        return data.cast_column(data, self.feature_name, self.feature)

class MarketClassLabelCast:
    def __init__(self, column_name: str) -> None:
        self.column_name: str = column_name

    def __call__(self, data: DatasetDict) -> DatasetDict:
        # get labels from all splits
        all_labels = sorted(set(
            label
            for split in data.values()
            for label in split[self.column_name]
        ))

        class_label = ClassLabel(
            num_classes=len(all_labels),
            names=list(map(str, all_labels)),
        )

        # cast for every split
        for split in data.keys():
            data[split] = data[split].cast_column(self.column_name, class_label)

        return data

class ClassLabelCast:
    def __init__(self, column_name: str) -> None:
        self.column_name: str = column_name

    def __call__(self, data: DatasetDict) -> DatasetDict:
        class_label = ClassLabel(
            num_classes=len(set(data["train"][self.column_name])),
            names=list(set(data["train"][self.column_name])),
        )

        return data.cast_column(
            self.column_name,
            class_label,
        )


class ToHFView:
    def __init__(
        self, name: str, id_column: str, features: Sequence[Feature], num_proc: int = 1
    ):
        self.name = name
        self.id_column = id_column
        self.features = features
        self.num_proc = num_proc

    def __call__(self, data: DatasetDict) -> DatasetView:
        if not all(
            self.id_column in split_columns
            for split_columns in data.column_names.values()
        ):
            data = data.map(
                lambda _, index: {self.id_column: index},
                with_indices=True,
                batched=True,
                num_proc=self.num_proc,
            )

        return HFDatasetView(
            name=self.name,
            hf_dataset=data,
            id_column=self.id_column,
            features=self.features,
        )


def imdb_process(data: DatasetDict, seed: int = 42):
    del data["unsupervised"]

    data = data.cast_column(
        "label",
        ClassLabel(
            num_classes=len(set(data["train"]["label"])),
            names=list(set(data["train"]["label"])),
        ),
    )

    return data


def imagenet_process(
    data: DatasetDict,
    seed: int = 42,
    num_proc: Optional[int] = None,
    batch_size: int = 1000,
):
    num_proc = num_proc or os.cpu_count() or 1
    del data["test"]
    train_data = data["train"].train_test_split(
        train_size=100_000, seed=seed, stratify_by_column="label"
    )["train"]
    data["train"] = train_data
    data["test"] = data["validation"]
    del data["validation"]

    imagenet_df = read_imagenet_labels()
    imagenet_df["synset_id"] = imagenet_df["pos"] + imagenet_df["offset"]

    def mapping(samples, indices: int):
        synset_ids = [
            imagenet_df.loc[imagenet_df["class_id"] == label, "synset_id"].item()
            for label in samples["label"]
        ]
        sample_ids = [
            f"{synset_id}_{index}" for synset_id, index in zip(synset_ids, indices)
        ]
        return {"synset_id": synset_ids, "sample_id": sample_ids}

    data = data.map(
        mapping,
        batched=True,
        with_indices=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    return data

def load_market1501(path):
    def parse_filename(fname):
        basename = os.path.basename(fname)
        pid_str = basename.split("_")[0]
        pid = int(pid_str)
        camid_str = basename.split("_")[1]
        camid = int(camid_str[1])
        return pid, camid

    def load_split(folder, split_name):
        records = []
        for img_path in glob.glob(os.path.join(path, folder, "*.jpg")):
            pid, camid = parse_filename(img_path)
            if pid == -1:
                continue  # skip junk
            records.append({
                "image": img_path,
                "pid_original": pid,
                "pid": str(pid),
                "camid": str(camid),
                "split": split_name
            })
        return records

    train_records   = load_split("bounding_box_train", "train")
    query_records   = load_split("query", "query")
    gallery_records = load_split("bounding_box_test", "gallery")

    all_records = train_records + query_records + gallery_records
    all_pids = sorted(set(r["pid"] for r in all_records))
    pid_map = {pid: str(i) for i, pid in enumerate(all_pids)}

    for r in all_records:
        r["pid"] = pid_map[r["pid"]]

    def to_dataset(records):
        features = Features({
            "image": Image(),
            "pid": Value("string"),
            "pid_original": Value("int32"),
            "camid": Value("string"),
            "split": Value("string"),
        })
        return Dataset.from_list(records, features=features)

    return DatasetDict({
        "train":   to_dataset([r for r in all_records if r["split"] == "train"]),
        "query":   to_dataset([r for r in all_records if r["split"] == "query"]),
        "gallery": to_dataset([r for r in all_records if r["split"] == "gallery"]),
    })