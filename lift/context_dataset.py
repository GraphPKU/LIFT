import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
)
from typing import List, Tuple
from copy import deepcopy


class ContextDataset(Dataset):
    """Given the context, `ContextDataset` creates a torch-Dataset, using the truncation strategy described in our
    paper.
    """
    def __init__(
        self,
        context: str,
        tokenizer: PreTrainedTokenizer,
        model_max_length: int = 7800,
        block_size: int = 256,
        len_segment: int = 8,
        len_offset: int = 3
    ):
        """
        Args:
            context (`str`):
                The context (e.g. an article).
            tokenizer (`transformers.PreTrainedTokenizer`):
                The tokenizer of the model to train.
            model_max_length (`int`, *optional*):
                The size of the context window of the model to train. The texts will be clipped to fit in the context
                window. Defaults to 7800.
            block_size (`int`, *optional*):
                The number of tokens in a block; a block is the unit of segments and offsets. Defaults to 256.
            len_segment (`int`, *optional*):
                The number of units in a segment; the article is divided into segments. Defaults to 8.
            len_offset (`int`, *optional*):
                The number of units per offset; it determines the offset between one segment and the next one.
                Defaults to 3.
        """
        self.ignore_index = -100  # The default value for ignored labels in torch
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        texts = context.replace('\0', ' ')
        input_ids = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        len_segment = len_segment * block_size
        len_offset = len_offset * block_size
        # Generate datapoints
        self.data = [(input_ids[s:s+len_segment], 0) for s in range(0, len(input_ids), len_offset)]
        self.num_segments = len(self.data)  # record the number of context datapoints

    def __len__(self):
        return self.num_segments

    def preprocessing(self, example: Tuple[List[int], int]):
        input_ids, len_input = example
        labels = deepcopy(input_ids)
        # Clip and truncation
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        # Transfer to Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[:len_input] = self.ignore_index  # mask the unsupervised part
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
    
    def __getitem__(self, index):
        return self.preprocessing(self.data[index])
    
    def enable_qa(self):
        raise NotImplementedError
    
    def disable_qa(self):
        raise NotImplementedError
    
    def generate_task(self):
        raise NotImplementedError
