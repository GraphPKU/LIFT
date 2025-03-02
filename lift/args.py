from dataclasses import dataclass, field
from typing import Optional, Any, List
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='',
        metadata={'help': "The HF name or the local path to the model."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': "The HF name or the local path to the tokenizer. It is the same as `model_name_or_path` by "
                  "default."}
    )
    model_max_length: Optional[int] = field(
        default=7800,
        metadata={'help': "The size of the context window of the model to train. The texts will be clipped to fit in "
                  "the context window."}
    )
    
    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    block_size: Optional[int] = field(
        default=256,
        metadata={'help': "The number of tokens in a block; a block is the unit of segments and offsets."}
    )
    len_segment: int = field(
        default=8,
        metadata={'help': "The number of units in a segment; the article is divided into segments."}
    )
    len_offset: int = field(
        default=3,
        metadata={'help': "The number of units per offset; it determines the offset between one segment and the next"
                  "one."}
    )


@dataclass
class CustomTrainingArguments:
    use_lora: bool = field(
        default=False,
        metadata={'help': "Use LoRA."}
    )
    lora_rank: Optional[int] = field(
        default=None,
        metadata={'help': "The LoRA rank. Required when `use_lora=True`."}
    )
    use_pissa: bool = field(
        default=False,
        metadata={'help': "Use PiSSA."}
    )
    use_gated_memory: bool = field(
        default=False,
        metadata={'help': "Use the gated-memory technique."}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={'help': "Use 4bit quantization."}
    )
    gather_batches: bool = field(
        default=False,
        metadata={'help': "Force the trainer to update the model only once every epoch. It is implemented with "
                  "gradient accumulation and it may lead to more stable gradients."}
    )
    involve_qa_epochs: int = field(
        default=0,
        metadata={'help': "The number of epochs of the 2nd stage."}
    )
    
    def __post_init__(self):
        if self.use_pissa:
            assert self.use_lora, "LoRA must be enabled when using PiSSA."
        assert int(self.use_gated_memory) + int(self.use_lora) <= 1, "LoRA and the gated-memory technique cannot be used simultaneously."


def parse_args(class_clusters: tuple[Any|tuple[Any]], no_dict: tuple[Any], return_config: bool=False):
    class_set = set()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            class_set.update(set(cluster))
        else:
            class_set.add(cluster)
    class_tuple = tuple(class_set)
    parser = HfArgumentParser(class_tuple)
    arg_list = parser.parse_args_into_dataclasses()
    arg_dict = {c: a for c, a in zip(class_tuple, arg_list)}
    returns = ()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            temp = {}
            for item in cluster:
                temp.update(dict(vars(arg_dict[item])))
            returns += (temp,)
        else:
            if cluster in no_dict:
                returns += (arg_dict[cluster],)
            else:
                returns += (dict(vars(arg_dict[cluster])),)
    if return_config:
        config = {}
        for arg in arg_list:
            config.update({k: v for k, v in dict(vars(arg)).items() if isinstance(v, int|float|bool|str)})
        return returns, config
    return returns
