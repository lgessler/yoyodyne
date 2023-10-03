"""LSTM model classes. test"""

from typing import Tuple

import torch
from torch import nn

from ... import data, defaults
from . import attention, base


class LSTMModule(base.BaseModule):
    """Base encoder for LSTM."""

    # Model arguments.
    bidirectional: bool
    # Constructed inside __init__.
    module: nn.LSTM

    def __init__(
        self,
        *args,
        bidirectional=defaults.BIDIRECTIONAL,
        **kwargs,
    ):
        """Initializes the encoder-decoder without attention.

        Args:
            *args: passed to superclass.
            bidirectional (bool).
            **kwargs: passed to superclass.
        """
        super().__init__(*args, **kwargs)
        self.bidirectional = bidirectional
        self.module = self.get_module()

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    def init_embeddings(
        self, num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        return self._normal_embedding_initialization(
            num_embeddings, embedding_size, pad_idx
        )


class LSTMEncoder(LSTMModule):
    def __init__(self, *args, **kwargs):
        self.tama_encoder_strategy = kwargs["tama_encoder_strategy"]
        if self.tama_encoder_strategy == "concat":
            kwargs["embedding_size"] *= 2
        super().__init__(*args, **kwargs)
        # We need to override the embedding init to account for the concat
        if self.tama_encoder_strategy == "concat":
            assert kwargs["embedding_size"] % 2 == 0
            self.embeddings = self.init_embeddings(
                kwargs["num_embeddings"],
                kwargs["embedding_size"] // 2,
                kwargs["pad_idx"]
            )

    def forward(
        self, 
        batch: data.PaddedBatch,
        projected_translation: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encodes the input.

        Args:
            source (data.PaddedTensor): source padded tensors and mask
                for source, of shape B x seq_len x 1.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                encoded timesteps, and the LSTM h0 and c0 cells.
        """
        source = batch.source
        embedded = self.embed(source.padded)

        if self.tama_encoder_strategy == "init_char":
            embedded = torch.concat((projected_translation.unsqueeze(1), embedded), dim=1)
        elif self.tama_encoder_strategy == "concat":
            expanded_translation = projected_translation.unsqueeze(1).repeat(1, embedded.shape[1], 1)
            embedded = torch.concat((embedded, expanded_translation), dim=2)
        # Packs embedded source symbols into a PackedSequence.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, source.lengths(), batch_first=True, enforce_sorted=False
        )
        # -> B x seq_len x encoder_dim, (h0, c0).
        if self.tama_encoder_strategy == "init_state":
            d0 = (2 if self.bidirectional else 1) * self.layers
            h = projected_translation.shape[-1]
            assert self.hidden_size % h == 0
            d2 = self.hidden_size // h
            packed_outs, (H, C) = self.module(
                packed,
                (
                    projected_translation.unsqueeze(0).repeat(d0, 1, d2),
                    projected_translation.unsqueeze(0).repeat(d0, 1, d2)
                )
            )
        else:
            packed_outs, (H, C) = self.module(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        return base.ModuleOutput(encoded, hiddens=(H, C))

    def get_module(self) -> nn.LSTM:
        return nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size * self.num_directions

    @property
    def name(self) -> str:
        return "LSTM"


class LSTMDecoder(LSTMModule):
    def __init__(self, *args, decoder_input_size, **kwargs):
        self.tama_decoder_strategy = kwargs["tama_decoder_strategy"]
        if self.tama_decoder_strategy == "concat":
            kwargs["embedding_size"] *= 2
        self.decoder_input_size = decoder_input_size
        super().__init__(*args, **kwargs)
        if self.tama_decoder_strategy == "concat":
            assert kwargs["embedding_size"] % 2 == 0
            self.embeddings = self.init_embeddings(
                kwargs["num_embeddings"],
                kwargs["embedding_size"] // 2,
                kwargs["pad_idx"]
            )

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        projected_translation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single decode pass.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]): last hidden
                states from the decoder of shape
                (1 x B x decoder_dim, 1 x B x decoder_dim).
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Decoder output,
                and the previous hidden states from the decoder LSTM.
        """
        embedded = self.embed(symbol)
        if self.tama_decoder_strategy == "concat":
            expanded_translation = projected_translation.unsqueeze(1).repeat(1, embedded.shape[1], 1)
            embedded = torch.concat((embedded, expanded_translation), dim=2)
        # -> 1 x B x decoder_dim.
        # Get the index of the last unmasked tensor.
        # -> B.
        last_encoder_out_idxs = (~encoder_mask).sum(dim=1) - 1
        # -> B x 1 x 1.
        last_encoder_out_idxs = last_encoder_out_idxs.view(
            encoder_out.size(0), 1, 1
        )
        # -> 1 x 1 x encoder_dim. This indexes the last non-padded dimension.
        last_encoder_out_idxs = last_encoder_out_idxs.expand(
            -1, -1, encoder_out.size(-1)
        )
        # -> B x 1 x encoder_dim.
        last_encoder_out = torch.gather(encoder_out, 1, last_encoder_out_idxs)
        # The input to decoder LSTM is the embedding concatenated to the
        # weighted, encoded, inputs.
        output, hiddens = self.module(
            torch.cat((embedded, last_encoder_out), 2), last_hiddens
        )
        output = self.dropout_layer(output)
        return base.ModuleOutput(output, hiddens=hiddens)

    def get_module(self) -> nn.LSTM:
        return nn.LSTM(
            self.decoder_input_size + self.embedding_size,
            self.hidden_size,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    @property
    def output_size(self) -> int:
        return self.num_embeddings

    @property
    def name(self) -> str:
        return "LSTM"


class LSTMAttentiveDecoder(LSTMDecoder):
    attention_input_size: int       
    def __init__(self, *args, attention_input_size, **kwargs):
        """Initializes the encoder-decoder with attention."""
        self.tama_decoder_strategy = kwargs["tama_decoder_strategy"]
        super().__init__(*args, **kwargs)
        self.attention_input_size = attention_input_size
        self.attention = attention.Attention(
            self.attention_input_size, self.hidden_size
        )

    def forward(
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        projected_translation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single decode pass.

        Args:
            symbol (torch.Tensor): previously decoded symbol of shape B x 1.
            last_hiddens (Tuple[torch.Tensor, torch.Tensor]): last hidden
                states from the decoder of shape
                (1 x B x decoder_dim, 1 x B x decoder_dim).
            encoder_out (torch.Tensor): encoded input sequence of shape
                B x seq_len x encoder_dim.
            encoder_mask (torch.Tensor): mask for the encoded input batch of
                shape B x seq_len.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Decoder output,,
                and the previous hidden states from the decoder LSTM.
        """
        #B x seq_len x embed_dim
        embedded = self.embed(symbol)
        if self.tama_decoder_strategy == "concat":
            expanded_translation = projected_translation.unsqueeze(1).repeat(1, embedded.shape[1], 1)
            embedded = torch.concat((embedded, expanded_translation), dim=2)
        last_h0, last_c0 = last_hiddens
        context, attention_weights = self.attention(
            last_h0.transpose(0, 1), encoder_out, encoder_mask
        )
        context.repeat(1, embedded.shape[1], 1)
        output, hiddens = self.module(
            torch.cat((embedded, context), 2), last_hiddens
        )
        output = self.dropout_layer(output)
        return base.ModuleOutput(output, hiddens=hiddens)

    @property
    def name(self) -> str:
        return "attentive LSTM"
