import re
import logging
from typing import Optional, Union, Collection, List, Dict
import numpy as np


logger = logging.getLogger(__name__)


def mask_by_candidates(
    logits: np.ndarray,
    candidates: Optional[Union[str, List[str]]],
    vocab: List[str],
    letter2id: Dict[str, int],
    ignored_tokens: List[int],
):
    if candidates is None:
        return logits

    _candidates = [letter2id[word] for word in candidates]
    _candidates.sort()
    _candidates = np.array(_candidates, dtype=int)

    candidates = np.zeros((len(vocab),), dtype=bool)
    candidates[_candidates] = True
    # candidates[-1] = True  # for cnocr, 间隔符号/填充符号，必须为真
    candidates[ignored_tokens] = True
    candidates = np.expand_dims(candidates, axis=(0, 1))  # 1 x 1 x (vocab_size+1)
    candidates = candidates.repeat(logits.shape[1], axis=1)

    masked = np.ma.masked_array(data=logits, mask=~candidates, fill_value=-100.0)
    logits = masked.filled()
    return logits


class CTCLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        cand_alphabet: Optional[Union[Collection, str]] = None,
    ):
        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = []
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

        self._candidates = None
        self.set_cand_alphabet(cand_alphabet)

    def set_cand_alphabet(self, cand_alphabet: Optional[Union[Collection, str]]):
        """
        设置待识别字符的候选集合。

        Args:
            cand_alphabet (Optional[Union[Collection, str]]): 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围

        Returns:
            None

        """
        if cand_alphabet is None:
            self._candidates = None
        else:
            cand_alphabet = [
                word if word != ' ' else '<space>' for word in cand_alphabet
            ]
            excluded = set([word for word in cand_alphabet if word not in self.dict])
            if excluded:
                logger.warning(
                    'chars in candidates are not in the vocab, ignoring them: %s'
                    % excluded
                )
            candidates = [word for word in cand_alphabet if word in self.dict]
            self._candidates = None if len(candidates) == 0 else candidates
            logger.debug('candidate chars: %s' % self._candidates)

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
    
    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        preds = mask_by_candidates(
            preds,
            self._candidates,
            self.character,
            self.dict,
            self.get_ignored_tokens(),
        )

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label
