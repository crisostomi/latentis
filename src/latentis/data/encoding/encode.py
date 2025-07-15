import functools
import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from latentis.benchmark.task import Task
from latentis.data import DATA_DIR
from latentis.data.dataset import Feature, HFDatasetView
from latentis.data.encoding.text_pooling import Pooler
from latentis.data.processor import DatasetView
from latentis.data.utils import default_collate
from latentis.nn import LatentisModule
from latentis.nn.encoders import TextHFEncoder
from latentis.space import Space
from latentis.space.vector_source import HDF5Source

pylogger = logging.getLogger(__name__)

DEFAULT_ENCODERS = {
    "text": [
        "bert-base-cased",
        "bert-base-uncased",
        "google/electra-base-discriminator",
        "roberta-base",
        "albert-base-v2",
        "xlm-roberta-base",
        "openai/clip-vit-base-patch32",
    ],
    "image": [
        "rexnet_100",
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "vit_base_resnet50_384",
        "vit_small_patch16_224",
        "openai/clip-vit-base-patch32",
        "efficientnet_b1_pruned",
        "regnety_002",
        "cspresnext50",
        "cspdarknet53",
    ],
}


class IdentityPooling(Pooler):
    def __init__(self, output_dim: int):
        super().__init__(name="no_pooling", output_dim=output_dim)

    def forward(self, x, **kwargs):
        return [(x, {})]


def _run(
    self,
):
    pylogger.info(f"Encoding {self.feature} for dataset {dataset._name}")

    model = self.model.to(self.device)

    split2pooler2space = defaultdict(dict)
    for split, split_data in dataset.hf_dataset.items():
        pooler2space = {
            pooler: Space(
                vector_source=HDF5Source(
                    shape=(len(split_data), pooler.output_dim),
                    root_dir=DATA_DIR
                    / dataset.name
                    / "encodings"
                    / split
                    / self.feature.name
                    / pooler.name,
                ),
                metadata={
                    **{
                        f"model/{key}": value for key, value in model.properties.items()
                    },
                    "feature": self.feature.name,
                    "split": split,
                    "dataset": dataset.name,
                    "pooler": pooler.__class__.__name__,
                },
            )
            for pooler in self.poolers
        }

        loader = DataLoader(
            split_data,
            batch_size=self.encoding_batch_size,
            pin_memory=self.device != torch.device("cpu"),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=functools.partial(
                self.collate_fn, model=model, feature=self.feature.name
            ),
        )

        for batch in tqdm(
            loader,
            desc=f"Encoding `{split}` samples for feature {self.feature.name} using {model.item_id[:8]}",
            total=len(loader),
        ):
            raw_encoding = model.encode(batch.to(self.device))

            for pooler in self.poolers:
                encoding2pooler_properties = pooler(**raw_encoding)

                for encoding, _pooler_properties in encoding2pooler_properties:
                    pooler2space[pooler].add_vectors(
                        vectors=encoding,
                        keys=batch[dataset._id_column].cpu().tolist(),
                        write=True,
                    )

        split2pooler2space[split] = pooler2space

    model.cpu()

    return split2pooler2space


@dataclass(frozen=True)
class EncodeResult:
    space: Space


class EncodeTask(Task):
    def __init__(
        self,
        dataset_view: DatasetView,
        split: str,
        feature: str,
        model: LatentisModule,
        encoding_batch_size: int,
        num_workers: int,
        device: torch.device,
        collate_fn: callable = default_collate,
        save_source_model: bool = False,
        pooler: Optional[nn.Module] = None,
        target_path: Optional[str] = None,
        write_every: int = 5,
        only_first_N_samples: Optional[int] = None,
    ):
        super().__init__()
        if split not in dataset_view.hf_dataset:
            raise ValueError(
                f"Split `{split}` not found in dataset `{dataset_view.name}`"
            )

        self.dataset = dataset_view
        self.feature: Feature = dataset_view.get_feature(feature)
        self.split = split
        self.model = model
        self.collate_fn = collate_fn
        self.encoding_batch_size = encoding_batch_size
        self.num_workers = num_workers
        self.device = device
        self.save_source_model = save_source_model
        self.pooler = pooler or IdentityPooling(output_dim=self.model.output_dim)
        self.target_path = target_path
        self.write_every: int = write_every
        self.only_first_N_samples = only_first_N_samples

    def metadata(self):
        return {
            "dataset": self.dataset.name,
            "split": self.split,
            "feature": self.feature.name,
            "model": self.model.hash,
            "collate_fn": self.collate_fn.__name__,
            "save_source_model": self.save_source_model,
            "pooler": self.pooler.__class__.__name__,
        }

    def _mean_pool(self, hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        sum_embeddings = torch.sum(hidden_state * mask, dim=1)
        sum_mask = mask.sum(dim=1)
        return sum_embeddings / sum_mask.clamp(min=1e-9)

    def _run(
        self,
    ) -> Space:
        pylogger.info(f"Encoding {self.feature} for dataset {self.dataset._name}")

        space_path = self.target_path or (
            DATA_DIR / self.dataset.name / "encodings" / self.hash
        )
        if space_path.exists():
            pylogger.info(f"Loading existing encodings from {space_path}")
            space = Space.load_from_disk(
                space_path, load_source_model=self.save_source_model
            )
            return EncodeResult(space=space)

        model_device = self.model.device

        model = self.model.to(self.device)

        split_data = self.dataset.hf_dataset[self.split]

        space = Space(
            vector_source=HDF5Source(
                shape=(len(split_data), self.pooler.output_dim),
                root_dir=space_path,
            ),
            metadata={
                **{f"model/{key}": value for key, value in model.metadata.items()},
                "feature": self.feature.name,
                "split": self.split,
                "dataset": self.dataset.name,
                "pooler": self.pooler.__class__.__name__,
            },
        )

        if self.only_first_N_samples is not None:
            split_data = split_data.select(
                range(self.only_first_N_samples)
            )

        loader = DataLoader(
            split_data,
            batch_size=self.encoding_batch_size,
            pin_memory=self.device != torch.device("cpu"),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=functools.partial(
                self.collate_fn, model=model, feature=self.feature.name
            ),
        )

        for i_batch, batch in tqdm(
            enumerate(loader),
            desc=f"Encoding `{self.split}` samples for feature {self.feature.name} using {model.hash}",
            total=len(loader),
        ):
            raw_encoding = model.encode(batch.to(self.device))

            if isinstance(raw_encoding['x'], (tuple)): # for AutoModels like BERT
                hidden_state = raw_encoding['x'][-1]
                attention_mask = raw_encoding['attention_mask']
                raw_encoding['x'] = self._mean_pool(hidden_state, attention_mask)

            encoding2pooler_properties = self.pooler(**raw_encoding)
            if len(encoding2pooler_properties) > 1:
                raise ValueError(
                    f"Multiple encodings returned by pooler `{self.pooler}`. Expected only one encoding per batch."
                )

            write: bool = i_batch % self.write_every == 0
            for encoding, _pooler_properties in encoding2pooler_properties:
                space.add_vectors(
                    vectors=encoding, keys=batch[self.dataset._id_column], write=write
                )

        space.save_to_disk(
            target_path=space_path,
            save_vector_source=True,
            save_metadata=True,
            save_source_model=self.save_source_model,
        )
        model.to(model_device)

        return EncodeResult(space=space)


if __name__ == "__main__":
    datasets = ["imagenet_text"]
    for dataset_name, hf_encoder in itertools.product(
        datasets,
        [
            # "FacebookAI/roberta-large",
            # "FacebookAI/roberta-base",
            # "google-bert/bert-base-uncased",
            # "google-bert/bert-base-cased",
            # "sentence-transformers/all-mpnet-base-v2",
            # "WinKawaks/vit-small-patch16-224",
            # "google/vit-base-patch16-224",
            # "google/vit-large-patch16-224",
            # "facebook/dinov2-base",
            # "facebook/dinov2-large",
            "google/vit-large-patch32-384",
            # "facebook/vit-mae-huge",
            # "google/vit-large-patch16-224-in21k",
            "openai/clip-vit-large-patch14",
        ],
    ):
        dataset = HFDatasetView.load_from_disk(DATA_DIR / dataset_name)

        for split in dataset.splits():
            encoder = TextHFEncoder(hf_encoder)
            # encoder = ImageHFEncoder(hf_encoder)
            task = EncodeTask(
                dataset_view=dataset,
                split=split,
                feature="text",
                model=encoder,
                collate_fn=default_collate,
                encoding_batch_size=128,
                num_workers=16,
                save_source_model=False,
                # pooler=HFPooler(layers=[encoder.num_layers - 1], pooling_fn=mean_pool, output_dim=encoder.output_dim),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                target_path=DATA_DIR
                / dataset_name
                / "encodings"
                / hf_encoder.replace("/", "-")
                / split,
                write_every=5,
            )

            print(task.metadata())
            task.run()
            # Space.load_from_disk()
