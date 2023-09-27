"""Collators and related utilities."""

import argparse
import dataclasses
from typing import List

import torch

from .. import defaults, util
from . import tama_batches, tama_datasets


class LengthError(Exception):
    pass


@dataclasses.dataclass
class TamaCollator:
    """Pads data."""

    pad_idx: int
    has_features: bool
    has_target: bool
    separate_features: bool
    features_offset: int
    max_source_length: int = defaults.MAX_SOURCE_LENGTH
    max_target_length: int = defaults.MAX_TARGET_LENGTH

    def _source_length_error(self, padded_length: int) -> None:
        """Callback function to raise the error when the padded length of the
        source batch is greater than the `max_source_length` allowed.

        Args:
            padded_length (int): The length of the the padded tensor.

        Raises:
            LengthError.
        """
        if padded_length > self.max_source_length:
            raise LengthError(
                f"The length of a source sample ({padded_length}) is greater "
                f"than the `--max_source_length` specified "
                f"({self.max_source_length})"
            )

    def _target_length_warning(self, padded_length: int) -> None:
        """Callback function to log a message when the padded length of the
        target batch is greater than the `max_target_length` allowed.

        Since `max_target_length` just truncates during inference, this is
        simply a suggestion.

        Args:
            padded_length (int): The length of the the padded tensor.
        """
        if padded_length > self.max_target_length:
            util.log_info(
                f"The length of a batch ({padded_length}) is greater than the "
                f"`--max_target_length` specified ({self.max_target_length}); "
                f"decoding at inference time will likely be truncated. "
                f"Consider increasing `--max_target_length`."
            )

    def concatenate_source_and_features(
        self,
        itemlist: List[tama_datasets.TamaItem],
    ) -> List[torch.Tensor]:
        """Concatenates source and feature tensors."""
        return [
            (
                torch.cat((item.source, item.features + self.features_offset))
                if item.has_features
                else item.source
            )
            for item in itemlist
        ]

    def pad_source(
        self, itemlist: List[tama_datasets.TamaItem]
    ) -> tama_batches.TamaPaddedTensor:
        """Pads source.

        Args:
            itemlist (List[tama_datasets.TamaItem]).

        Returns:
            tama_batches.TamaPaddedTensor.
        """
        return tama_batches.TamaPaddedTensor(
            [item.source for item in itemlist],
            self.pad_idx,
            self._source_length_error,
        )

    def pad_source_features(
        self,
        itemlist: List[tama_datasets.TamaItem],
    ) -> tama_batches.TamaPaddedTensor:
        """Pads concatenated source and features.

        Args:
            itemlist (List[tama_datasets.TamaItem]).

        Returns:
            tama_batches.TamaPaddedTensor.
        """
        return tama_batches.TamaPaddedTensor(
            self.concatenate_source_and_features(itemlist),
            self.pad_idx,
            self._source_length_error,
        )

    def pad_features(
        self,
        itemlist: List[tama_datasets.TamaItem],
    ) -> tama_batches.TamaPaddedTensor:
        """Pads features.

        Args:
            itemlist (List[tama_datasets.TamaItem]).

        Returns:
            tama_batches.TamaPaddedTensor.
        """
        return tama_batches.TamaPaddedTensor(
            [item.features for item in itemlist], self.pad_idx
        )

    def pad_target(
        self, itemlist: List[tama_datasets.TamaItem]
    ) -> tama_batches.TamaPaddedTensor:
        """Pads target.

        Args:
            itemlist (List[tama_datasets.TamaItem]).

        Returns:
            tama_batches.TamaPaddedTensor.
        """
        return tama_batches.TamaPaddedTensor(
            [item.target for item in itemlist],
            self.pad_idx,
            self._target_length_warning,
        )

    def __call__(self, itemlist: List[tama_datasets.TamaItem]) -> tama_batches.TamaPaddedBatch:
        """Pads all elements of an itemlist.

        Args:
            itemlist (List[tama_datasets.TamaItem]).

        Returns:
            tama_batches.TamaPaddedBatch.
        """
        padded_target = self.pad_target(itemlist) if self.has_target else None
        padded_translation_tensors = tama_batches.TamaPaddedTensor(
            [item.translation_tensors for item in itemlist], 0
        )
        if self.separate_features:
            return tama_batches.TamaPaddedBatch(
                self.pad_source(itemlist),
                features=self.pad_features(itemlist),
                target=padded_target,
                translation_tensors=padded_translation_tensors,
            )
        else:
            return tama_batches.TamaPaddedBatch(
                self.pad_source_features(itemlist),
                target=padded_target,
                translation_tensors=padded_translation_tensors,
            )

    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds collator options to the argument parser.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--max_source_length",
            type=int,
            default=defaults.MAX_SOURCE_LENGTH,
            help="Maximum source string length. Default: %(default)s.",
        )
        parser.add_argument(
            "--max_target_length",
            type=int,
            default=defaults.MAX_TARGET_LENGTH,
            help="Maximum target string length. Default: %(default)s.",
        )
