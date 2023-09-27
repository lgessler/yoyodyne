"""TSV parsing.

The TsvParser yields data from TSV files using 1-based indexing and custom
separators.
"""


import dataclasses
from typing import Iterator, List, Tuple, Union, Dict, Any

import jsonlines
import torch


class Error(Exception):
    """Module-specific exception."""
    pass


@dataclasses.dataclass
class TamaParser:
    """Streams data from a jsonlines file."""

    @staticmethod
    def _jsonl_reader(path: str) -> Iterator[str]:
        with jsonlines.open(path, "r") as f:
            for x in f:
                yield x

    @property
    def has_features(self) -> bool:
        return False

    @property
    def has_target(self) -> bool:
        return True

    def samples(
        self, path: str
    ) -> Iterator[
        Tuple[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Yields source, and features and/or target if available."""
        assert path.endswith("_tama.jsonl")
        tensor_path = path.replace("_tama.jsonl", ".pt")
        with open(tensor_path, 'rb') as f:
            tensors = torch.load(f)
        for row, tensor in zip(self._jsonl_reader(path), tensors):
            yield row, tensor

    # String parsing methods.

    @staticmethod
    def _get_symbols(string: str, sep: str) -> List[str]:
        return list(string) if not sep else string.split(sep)

    def source_symbols(self, string: str) -> List[str]:
        return self._get_symbols(string, "")

    def features_symbols(self, string: str) -> List[str]:
        # We deliberately obfuscate these to avoid overlap with source.
        return [
            f"[{symbol}]"
            for symbol in self._get_symbols(string, ";")
        ]

    def target_symbols(self, string: str) -> List[str]:
        return self._get_symbols(string, "")

    # Deserialization methods.

    def source_string(self, symbols: List[str]) -> str:
        return "".join(symbols)

    def features_string(self, symbols: List[str]) -> str:
        return ";".join(
            # This indexing strips off the obfuscation.
            [symbol[1:-1] for symbol in symbols],
        )

    def target_string(self, symbols: List[str]) -> str:
        return "".join(symbols)
