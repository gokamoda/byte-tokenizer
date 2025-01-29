# Copyright 2021 T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization class for model ByT5."""

import warnings
from itertools import combinations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ByteLMTokenizerV2(PreTrainedTokenizer):
    """Byte tokenizer with completely seperate space for special tokens.

    Parameters
    ----------
    PreTrainedTokenizer : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """

    model_input_names: list[str] = ["input_ids", "attention_mask"]
    reserve_sizes: list[int] = [25, 8, 8, 1]
    byte_head_ints: list[int] = [
        int("11100000", base=2),
        int("11000000", base=2),
        int("10000000", base=2),
        int("00000000", base=2),
    ]
    byte_n_free_bits: list[int] = [5, 5, 6, 7]
    patch_padding: bool
    reserve_token_list: list[tuple[int]]

    def __init__(
        self,
        patch_padding=False,
        pad_token: tuple= (0, "<pad>"),
        eos_token: tuple =(1, "</s>"),
        bos_token: tuple =(2, "<s>"),
        cls_token: tuple= (3, "<cls>"),
        sep_token: tuple= (4, "<sep>"),
        mask_token: tuple= (5, "<mask>"),
        **kwargs,
    ) -> None:
        assert np.all(
            np.array(self.reserve_sizes) > 0
        ), "Each byte must have at least one reserve index"

        assert np.prod(
            [
                2**n_free_bits - reserve_size
                for reserve_size, n_free_bits in zip(
                    self.reserve_sizes, self.byte_n_free_bits
                )
            ]
        ) >= int(
            "110000", base=16
        ), "Not enough positions for all unicode. Too many reserve size."

        self.patch_padding = patch_padding

        # list up all reserve tokens
        self._list_up_reserve_tokens()

        _bos_token = (
            AddedToken(bos_token[1], lstrip=False, rstrip=False)
            if isinstance(bos_token[1], str)
            else bos_token[1]
        )
        _eos_token = (
            AddedToken(eos_token[1], lstrip=False, rstrip=False)
            if isinstance(eos_token[1], str)
            else eos_token[1]
        )
        _pad_token = (
            AddedToken(pad_token[1], lstrip=False, rstrip=False)
            if isinstance(pad_token[1], str)
            else pad_token[1]
        )
        _cls_token = (
            AddedToken(cls_token[1], lstrip=False, rstrip=False)
            if isinstance(cls_token[1], str)
            else cls_token
        )
        _sep_token = (
            AddedToken(sep_token[1], lstrip=False, rstrip=False)
            if isinstance(sep_token[1], str)
            else sep_token[1]
        )
        _mask_token = (
            AddedToken(mask_token[1], lstrip=False, rstrip=False)
            if isinstance(mask_token[1], str)
            else mask_token[1]
        )

        self.offset = 0

        self._added_tokens_decoder = {
            self.reserve_token_list[pad_token[0]]: _pad_token,
            self.reserve_token_list[eos_token[0]]: _eos_token,
            self.reserve_token_list[bos_token[0]]: _bos_token,
            self.reserve_token_list[cls_token[0]]: _cls_token,
            self.reserve_token_list[sep_token[0]]: _sep_token,
            self.reserve_token_list[mask_token[0]]: _mask_token,
        }
        super().__init__(
            bos_token=_bos_token,
            eos_token=_eos_token,
            pad_token=_pad_token,
            cls_token=_cls_token,
            sep_token=_sep_token,
            mask_token=_mask_token,
            extra_ids=0,
            **kwargs,
        )

        self._vocab_size = len(self.get_vocab())

    def create_tree(
        self, byte_options: list[list[int]], byte_index: int, max_byte_index: int
    ) -> list[list[int]]:
        if byte_index == max_byte_index:
            return [[reserve_option] for reserve_option in byte_options[byte_index]]

        concat_list = []
        for byte_reserve_option in byte_options[byte_index]:
            if byte_reserve_option is not None:
                concat_list += [
                    [byte_reserve_option] + following_bytes
                    if following_bytes != [None]
                    else [byte_reserve_option]
                    for following_bytes in self.create_tree(
                        byte_options=byte_options,
                        byte_index=byte_index + 1,
                        max_byte_index=max_byte_index,
                    )
                ]
            else:
                concat_list.append([None])
        return concat_list

    def _list_up_reserve_tokens(self):
        byte_reserve_options = [
            list(range(reserve_size)) for reserve_size in self.reserve_sizes
        ]

        if not self.patch_padding:
            for i in range(len(byte_reserve_options) - 1):
                byte_reserve_options[i] += [None]

        byte_reserve_options.reverse()

        reserve_tokens = self.create_tree(
            byte_options=byte_reserve_options, byte_index=0, max_byte_index=3
        )

        reserve_tokens = sorted(
            reserve_tokens,
            key=lambda lst: sum([e * (256**i) for i, e in enumerate(lst)])
            + 256 ** len(lst),
        )
        for reserve_token_index in range(len(reserve_tokens)):
            reserve_tokens[reserve_token_index].reverse()
            for position in range(len(reserve_tokens[reserve_token_index])):
                reserve_tokens[reserve_token_index][position] += self.byte_head_ints[
                    position
                ]
            reserve_tokens[reserve_token_index] = tuple(
                reserve_tokens[reserve_token_index]
            )

        self.reserve_token_list = reserve_tokens

    @property
    def vocab_size(self):
        return self._vocab_size

    def get_vocab(self):
        byte_options = [
            list(range(reserve_size, 2**n_free_bits))
            for reserve_size, n_free_bits in zip(
                self.reserve_sizes, self.byte_n_free_bits
            )
        ]

        if not self.patch_padding:
            for i in range(len(byte_options) - 1):
                byte_options[i] += [None]

        byte_options.reverse()
        byte_tokens = self.create_tree(
            byte_options=byte_options, byte_index=0, max_byte_index=3
        )

        byte_tokens = sorted(
            byte_tokens,
            key=lambda lst: sum([e * (256**i) for i, e in enumerate(lst)])
            + 256 ** len(lst),
        )

        for byte_token_index in range(len(byte_tokens)):
            byte_tokens[byte_token_index].reverse()
            for position in range(len(byte_tokens[byte_token_index])):
                byte_tokens[byte_token_index][position] += self.byte_head_ints[position]
            byte_tokens[byte_token_index] = tuple(byte_tokens[byte_token_index])

        vocab = {self.convert_ids_to_tokens(tokens): tokens for tokens in byte_tokens}
        vocab.pop("")

        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: list[int]) -> list[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + list(self.eos_token_id)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in `padding_side` argument:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and len(required_input) != max_length
        )

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            if self.patch_padding:
                difference = (max_length - len(required_input)) // len(
                    self.byte_head_ints
                )
                mask_patch_size = 4
            else:
                difference = max_length - len(required_input)
                mask_patch_size = 1

            padding_side = (
                padding_side if padding_side is not None else self.padding_side
            )

            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = (
                        encoded_inputs["attention_mask"]
                        + [0] * difference * mask_patch_size
                    )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"]
                        + list(self.pad_token_type_id) * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = (
                        encoded_inputs["special_tokens_mask"]
                        + [1] * difference * mask_patch_size
                    )
                encoded_inputs[self.model_input_names[0]] = (
                    required_input + list(self.pad_token_id) * difference
                )
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [
                        0
                    ] * difference * mask_patch_size + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        list(self.pad_token_type_id) * difference
                        + encoded_inputs["token_type_ids"]
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference * mask_patch_size + encoded_inputs[
                        "special_tokens_mask"
                    ]
                encoded_inputs[self.model_input_names[0]] = (
                    list(self.pad_token_id) * difference + required_input
                )
            else:
                raise ValueError(f"Invalid padding strategy:{padding_side}")

        return encoded_inputs

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not
        make use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:
        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def _tokenize(self, text: str) -> list[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        token_ids = []
        for c in text:
            token_ids.extend(self.unicode_to_bytes(ord(c)))

        # Convert to string
        token_ids = [str(i) for i in token_ids]
        return token_ids

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        token_id = int(token) + self.offset
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return str(index - self.offset)

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self._added_tokens_encoder:
            return list(self._added_tokens_encoder[token])
        return [self._convert_token_to_id(token)]

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.extend(self._convert_token_to_id_with_added_voc(token))
        return ids

    def convert_bytes_for_single_char_to_char(self, ids: list[int]) -> str:
        byte_ints = []
        byte_offset = 1
        for byte_position in range(1, len(ids) + 1):
            if byte_position == 1 and ids[-byte_position] == 0:  # special token
                return self.added_tokens_decoder[tuple(ids)].__str__()

            byte_int = (
                ids[-byte_position]
                - self.byte_head_ints[-byte_position]
                - self.reserve_sizes[-byte_position]
            )
            if byte_int != -self.reserve_sizes[-byte_position]:  # not padding
                byte_ints.append(byte_int * byte_offset)

            byte_offset *= (
                2 ** self.byte_n_free_bits[-byte_position]
                - self.reserve_sizes[-byte_position]
            )

        codepoint = sum(byte_ints)
        if codepoint >= int("110000", base=16):
            return None
        else:
            return chr(codepoint)

    def convert_ids_to_tokens(
        self, ids: list[int] | tuple[int], skip_special_tokens: bool = False
    ) -> str | None:
        """convert ids for single/multiple unicode character(s) to unicode character(s)"""

        def is_special_token(ids: list[int]):
            return ids[-1] < self.reserve_sizes[-1]

        decoded_chars = ""

        if isinstance(ids, tuple):
            ids = list(ids)

        if self.patch_padding:
            for byte_position in range(0, len(ids), len(self.byte_head_ints)):
                char_bytes = ids[
                    byte_position : byte_position + len(self.byte_head_ints)
                ]
                if (
                    skip_special_tokens and not is_special_token(char_bytes)
                ) or not skip_special_tokens:
                    char = self.convert_bytes_for_single_char_to_char(char_bytes)
                    if char:
                        decoded_chars += self.convert_bytes_for_single_char_to_char(
                            char_bytes
                        )
            return decoded_chars

        if ids[-1] >= self.reserve_sizes[-1]:  # not special token
            byte_ints = []
            byte_offset = 1
            for byte_position in range(1, len(ids) + 1):
                if ids[-byte_position] == 0:
                    break
                byte_int = (
                    ids[-byte_position]
                    - self.byte_head_ints[-byte_position]
                    - self.reserve_sizes[-byte_position]
                )
                assert byte_int >= 0
                byte_ints.append(byte_int * byte_offset)
                byte_offset *= (
                    2 ** self.byte_n_free_bits[-byte_position]
                    - self.reserve_sizes[-byte_position]
                )

            codepoint = sum(byte_ints)
            if codepoint >= int("110000", base=16):
                return None
            else:
                return chr(codepoint)
        else:  # special token
            return self._added_tokens_decoder[tuple(ids)]

    def unicode_to_bytes(self, codepoint: int) -> list[int]:
        byte_list_reversed = []
        for byte_position_from_right in range(len(self.byte_n_free_bits)):
            byte_n_free_ids = (
                2 ** self.byte_n_free_bits[-1 - byte_position_from_right]
                - self.reserve_sizes[-1 - byte_position_from_right]
            )
            byte_id = (
                codepoint % byte_n_free_ids
                + self.reserve_sizes[-1 - byte_position_from_right]
                + self.byte_head_ints[-1 - byte_position_from_right]
            )
            codepoint //= byte_n_free_ids
            byte_list_reversed.append(byte_id)

            if codepoint == 0:
                if self.patch_padding:
                    for pad_byte_position_from_right in range(
                        len(byte_list_reversed), len(self.byte_n_free_bits)
                    ):
                        byte_list_reversed.append(
                            self.byte_head_ints[-1 - pad_byte_position_from_right]
                        )
                byte_list_reversed.reverse()
                return byte_list_reversed
        raise ValueError("codepoint is too large")

    # ByteTokenizer has no vocab file
    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        return ()


if __name__ == "__main__":
    tokenizer = ByteLMTokenizerV2(patch_padding=True, padding_side="left",
        pad_token=(0, "<pad>"),
        bos_token=(73, "<s>"),
        eos_token=(146, "</s>"),
        mask_token=(219, "<mask>"),
    )
    prompts = ["a", "aあ", "aあ\U00080000"]
    print(prompts)
    encoded = tokenizer(prompts, padding="longest")
    print(encoded)
    decoded = tokenizer.batch_decode(encoded["input_ids"])
    print(decoded)
    decoded_no_special = tokenizer.batch_decode(
        encoded["input_ids"], skip_special_tokens=True
    )
    print(decoded_no_special)

    from IPython import embed
    embed()
