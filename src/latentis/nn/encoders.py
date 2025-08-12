from typing import Optional, Sequence

import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
)

from latentis.nn._base import WrappedModule
from latentis.types import Metadata
from sentence_transformers import SentenceTransformer


class HFEncoder(WrappedModule):
    def __init__(
        self,
        hf_name: str,
        requires_grad: bool,
        encode_fn: Optional[str] = None,
        decode_fn: Optional[str] = None,
        metadata: Optional[Metadata] = None,
    ):
        if hf_name in ["sentence-transformers/clip-ViT-B-32"]:
            hf_model = SentenceTransformer(hf_name)
            # For compatibility
            hf_model.eval()
            hf_model.requires_grad_(requires_grad)
        else:
            hf_model: PreTrainedModel = (
                AutoModel.from_pretrained(
                    hf_name, output_hidden_states=True, return_dict=True
                )
                .eval()
                .requires_grad_(requires_grad)
            )
        self.hf_name = hf_name
        super().__init__(
            model=hf_model,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            metadata={**(metadata or {}), "hf_name": hf_name},
        )

    @property
    def output_dim(self):
        raise NotImplementedError


class TextHFEncoder(HFEncoder):
    def __init__(
        self,
        hf_name: str,
        requires_grad: bool = False,
        truncation: bool = True,
        padding: bool = True,
        max_length: Optional[int] = None,
        metadata: Optional[Metadata] = None,
        **kwargs,
    ):
        super().__init__(
            hf_name, requires_grad, encode_fn=None, decode_fn=None, metadata=metadata
        )
        if hf_name in ["sentence-transformers/clip-ViT-B-32"]: # currently fixing this
            self.tokenizer = self.model.tokenizer
            self.uses_sentence_transformer = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            self.uses_sentence_transformer = False

        if hasattr(self.model, "config"):
            max_length = max_length or self.model.config.max_length
        else:
            # SentenceTransformer fallback: use tokenizer's declared max length
            max_length = getattr(self.tokenizer, "model_max_length", 512)


        self.pre_encode_kwargs = {
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
            **kwargs,
        }

        self.is_clip: bool = "clip" in self.hf_name

        if hasattr(self.model, "config"):
            self._output_dim = (
                self.model.config.text_config.projection_dim
                if self.is_clip
                else self.model.config.hidden_size
            )
        else:
            # For SentenceTransformer
            self._output_dim = self.model.get_sentence_embedding_dimension()
            if self._output_dim is None:
                # fallback: infer from dummy input
                dummy_output = self.model.encode(["test"], convert_to_tensor=True)
                self._output_dim = dummy_output.shape[-1]

    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers

    @property
    def output_dim(self):
        return self._output_dim

    @torch.no_grad()
    def pre_encode(self, samples: Sequence, feature: str) -> BatchEncoding:
        texts = [sample[feature] for sample in samples]

        if self.uses_sentence_transformer:
            return BatchEncoding(dict(raw_text=texts))
        
        tok_out: BatchEncoding = self.tokenizer(
            texts,
            return_special_tokens_mask=True,
            return_token_type_ids=not self.is_clip,
            return_tensors="pt",
            **self.pre_encode_kwargs,
        )

        return BatchEncoding(dict(tok_out=tok_out))
    
    def _mean_pool(self, hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        sum_embeddings = torch.sum(hidden_state * mask, dim=1)
        sum_mask = mask.sum(dim=1)
        return sum_embeddings / sum_mask.clamp(min=1e-9)

    def encode(self, x: BatchEncoding):
        if self.uses_sentence_transformer:
            texts = x["raw_text"]
            embeddings = torch.tensor(self.model.encode(texts, convert_to_tensor=True))
            return {"x": embeddings, "mask": torch.ones(embeddings.size(0), dtype=torch.bool)}
        
        tok_out = x["tok_out"]

        mask = (
            tok_out["attention_mask"]
            * tok_out["special_tokens_mask"].bool().logical_not()
        )
        del tok_out["special_tokens_mask"]

        if "token_type_ids" in tok_out and self.hf_name.__contains__(("t5")):
            del tok_out["token_type_ids"]

        if self.hf_name.startswith("openai/clip"):
            # TODO: check this
            encodings = self.model.text_model(**tok_out, return_dict=True).pooler_output
        elif self.hf_name.startswith(("t5", "sentence-transformers/gtr")):
            # T5-style models
            encoder_output = self.model.encoder(**tok_out, return_dict=True).last_hidden_state
            attention_mask = tok_out["attention_mask"]
            encodings = self._mean_pool(encoder_output, attention_mask)
        else:
            encodings = self.model(**tok_out)["hidden_states"]

        return {"x": encodings, "mask": mask, "attention_mask": tok_out["attention_mask"]}


class ImageHFEncoder(HFEncoder):
    def __init__(
        self,
        hf_name: str,
        requires_grad: bool = False,
        metadata: Optional[Metadata] = None,
    ):
        super().__init__(hf_name, requires_grad, metadata=metadata)
        self.processor = AutoImageProcessor.from_pretrained(self.hf_name)

        self.is_clip: bool = "clip" in self.hf_name
        self.is_resnet: bool = "resnet" in self.hf_name.lower() or "convnext" in self.hf_name.lower()

        self._output_dim = (
            self.model.config.vision_config.projection_dim
            if self.is_clip
            else
            self.model.config.hidden_sizes[-1] if self.is_resnet # taking 2048
            else self.model.config.hidden_size
        )

    @property
    def output_dim(self):
        return self._output_dim

    @torch.no_grad()
    def pre_encode(self, samples: Sequence, feature: str, **kwargs):
        images = [sample[feature].convert("RGB") for sample in samples]
        # images = [sample[feature] for sample in samples]
        images = self.processor(images=images, return_tensors="pt")

        return {"proc_out": images}

    @torch.no_grad()
    def encode(self, x: BatchEncoding):
        x = x["proc_out"]
        if not self.is_clip and not self.is_resnet:
            outputs = self.model(**x)
            return {"x": outputs.last_hidden_state[:, 0, :]}
        elif self.is_resnet:
            out = self.model(**x, return_dict=True)
            pooled = getattr(out, "pooler_output", None)
            return {"x": pooled.squeeze(-1).squeeze(-1)}
        else:
            return {"x": self.model.get_image_features(**x)}
