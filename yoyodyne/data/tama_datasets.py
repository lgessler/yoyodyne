"""Datasets and related utilities.

Anything which has a tensor member should inherit from nn.Module, run the
superclass constructor, and register the tensor as a buffer. This enables the
Trainer to move them to the appropriate device."""

import dataclasses

from typing import Iterator, List, Optional, Any

import torch
from torch import nn
from torch.utils import data

from .. import special

from . import indexes, tama_parser


class TamaItem(nn.Module):
    """Source tensor, with optional features and target tensors.

    This represents a single item or observation."""

    source: torch.Tensor
    features: Optional[torch.Tensor]
    target: Optional[torch.Tensor]
    translation_wps: Optional[torch.Tensor]
    translation_tensors: Optional[torch.Tensor]

    def __init__(self, source, features=None, target=None, translation_wps=None, translation_tensors=None):
        """Initializes the item.

        Args:
            source (torch.Tensor).
            features (torch.Tensor, optional).
            target (torch.Tensor, optional).
        """
        super().__init__()
        self.register_buffer("source", source)
        self.register_buffer("features", features)
        self.register_buffer("target", target)
        self.register_buffer("translation_wps", translation_wps)
        self.register_buffer("translation_tensors", translation_tensors)

    @property
    def has_features(self):
        return self.features is not None

    @property
    def has_target(self):
        return self.target is not None


@dataclasses.dataclass
class TamaDataset(data.Dataset):
    """Datatset class."""

    samples: List[Any]
    index: indexes.Index  # Usually copied from the DataModule.
    parser: tama_parser.TamaParser  # Ditto.

    @property
    def has_features(self) -> bool:
        return self.parser.has_features

    @property
    def has_target(self) -> bool:
        return self.parser.has_target

    def _encode(
        self,
        symbols: List[str],
        symbol_map: indexes.SymbolMap,
    ) -> torch.Tensor:
        """Encodes a sequence as a tensor of indices with string boundary IDs.

        Args:
            symbols (List[str]): symbols to be encoded.
            symbol_map (indexes.SymbolMap): symbol map to encode with.

        Returns:
            torch.Tensor: the encoded tensor.
        """
        return torch.tensor(
            [
                symbol_map.index(symbol, self.index.unk_idx)
                for symbol in symbols
            ],
            dtype=torch.long,
        )

    def encode_source(self, symbols: List[str]) -> torch.Tensor:
        """Encodes a source string, padding with start and end tags.

        Args:
            symbols (List[str]).

        Returns:
            torch.Tensor.
        """
        wrapped = [special.START]
        wrapped.extend(symbols)
        wrapped.append(special.END)
        return self._encode(wrapped, self.index.source_map)

    def encode_features(self, symbols: List[str]) -> torch.Tensor:
        """Encodes a features string.

        Args:
            symbols (List[str]).

        Returns:
            torch.Tensor.
        """
        return self._encode(symbols, self.index.features_map)

    def encode_target(self, symbols: List[str]) -> torch.Tensor:
        """Encodes a features string, padding with end tags.

        Args:
            symbols (List[str]).

        Returns:
            torch.Tensor.
        """
        wrapped = symbols
        wrapped.append(special.END)
        return self._encode(wrapped, self.index.target_map)

    # Decoding.

    def _decode(
        self,
        indices: torch.Tensor,
        symbol_map: indexes.SymbolMap,
    ) -> Iterator[List[str]]:
        """Decodes the tensor of indices into lists of symbols.

        Args:
            indices (torch.Tensor): 2d tensor of indices.
            symbol_map (indexes.SymbolMap).

        Yields:
            List[str]: Decoded symbols.
        """
        for idx in indices.cpu().numpy():
            yield [
                symbol_map.symbol(c)
                for c in idx
                if c not in self.index.special_idx
            ]

    def decode_source(
        self,
        indices: torch.Tensor,
    ) -> Iterator[str]:
        """Decodes a source tensor.

        Args:
            indices (torch.Tensor): 2d tensor of indices.

        Yields:
            str: Decoded source strings.
        """
        for symbols in self._decode(indices, self.index.source_map):
            yield self.parser.source_string(symbols)

    def decode_features(
        self,
        indices: torch.Tensor,
    ) -> Iterator[str]:
        """Decodes a features tensor.

        Args:
            indices (torch.Tensor): 2d tensor of indices.

        Yields:
            str: Decoded features strings.
        """
        for symbols in self._decode(indices, self.index.target_map):
            yield self.parser.features_string(symbols)

    def decode_target(
        self,
        indices: torch.Tensor,
    ) -> Iterator[str]:
        """Decodes a target tensor.

        Args:
            indices (torch.Tensor): 2d tensor of indices.

        Yields:
            str: Decoded target strings.
        """
        for symbols in self._decode(indices, self.index.target_map):
            yield self.parser.target_string(symbols)

    # Required API.

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TamaItem:
        """Retrieves item by index.

        Args:
            idx (int).

        Returns:
            Item.
        """
        assert not self.has_features and self.has_target
        inst, [pool_repr, translation_tensors] = self.samples[idx]
        translation_wps = inst["translation_aligned_wps"]
        return TamaItem(
            source=self.encode_source(list(inst["source"])),
            target=self.encode_target(list(inst["target"])),
            translation_wps=torch.tensor(translation_wps),
            translation_tensors=translation_tensors,
        )
