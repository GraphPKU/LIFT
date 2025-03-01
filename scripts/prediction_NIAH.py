import math
import json
import numpy as np
import os
import pickle
import sys
import torch
import torch.utils
import torch.utils.data
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from glob import glob
from lift.context_dataset import ContextDataset
from lift.model import load_tokenizer, load_model
from lift.train import train
from lift.args import (
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments,
    parse_args
)
from nltk.tokenize import sent_tokenize
from numpy.random import randint
from pathlib import Path
from tqdm import tqdm
from transformers import TrainingArguments, PreTrainedTokenizer
from transformers.utils import logging
from typing import List, Dict, Optional, Tuple


sys.path.append(str(Path(__file__).parent.parent))
logger = logging.get_logger(__name__)


NIAHFORMAT_EN = """There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.
{context}
{prompt}
"""


class ICLContextDataset(ContextDataset):
    """Implement contextualized training introduced in our paper (3.3).
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
        mixin = self.tokenizer("...", add_special_tokens=False)['input_ids']
        LIFT_ICL_PROMPT = "Given above context, please recite following segment of the context: \n\n\n"
        prompt = self.tokenizer(LIFT_ICL_PROMPT, add_special_tokens=False)['input_ids']

        self.data = []
        for s in range(0, len(input_ids), len_offset):
            start_pos = s
            end_pos = min(s + len_segment, len(input_ids))
            lift_icl_front, lift_icl_back = self.prepare_lift_icl(start_pos, end_pos, input_ids, len_segment, len_offset)
            lift_icl = lift_icl_front + mixin + lift_icl_back

            lift_icl += prompt 
            input_len = len(lift_icl)
            lift = input_ids[start_pos: end_pos]

            self.data.append((lift_icl + lift, input_len))

        self.num_segments = len(self.data)  # record the number of context datapoints

    def __len__(self):
        return self.num_segments
    
    def prepare_lift_icl(self, start_pos: int, end_pos: int, input_ids: list[int], len_segment: int, len_offset: int, len_lift_icl: int=4096):

        def get_fix_length_segments(front_lim: int, back_lim: int, tot_len: int):
            """
                return two segments,
                satisfying frist segment length is less than 'front_lim', 
                second segment length is less than 'back_lim'
                and the sum of two segments' length is equal to 'tot_len'.


                The return value is the length of the two segments.
            """
            a = randint(max(0, tot_len - back_lim), min(front_lim, tot_len))
            b = tot_len - a
            return a, b

        front_len, back_len = get_fix_length_segments(len_lift_icl, len_lift_icl, len_lift_icl)
        front_st = randint(0, len_lift_icl - front_len)
        back_ed = randint(0, len_lift_icl - back_len)

        return input_ids[front_st: front_st+front_len], input_ids[-back_ed-back_len:-back_ed] if back_ed != 0 else input_ids[-back_len:]


@dataclass
class NIAHArgs:
    haystack_path: str = field(default="long-llm:needle/PaulGrahamEssays", metadata={'help': 'The context for evaluation.'})
    cache_input_path: Optional[str] = field(default=None, metadata={'help': "The path to the cached input file."})
    output_path: Optional[str] = field(default=None, metadata={'help': "The output file."})
    test_length_min: Optional[int] = field(default=None, metadata={'help': "The minimum length of the input."})
    test_length_max: Optional[int] = field(default=None,metadata={'help': "The maximum length of the input."})
    test_length_num: Optional[int] = field(default=None,metadata={'help': "The number of the tested input lengths."})
    test_length: List[int] = field(default_factory=lambda: [], metadata={'help': 'Specified evaluation lengths.'})
    test_depth_min: Optional[int] = field(default=None, metadata={'help': "The minimum depth of the needle (from 0 to 100)."})
    test_depth_max: Optional[int] = field(default=None, metadata={'help': "The maximum depth of the needle (from 0 to 100)."})
    test_depth_num: Optional[int] = field(default=None, metadata={'help': "The number of the tested needle depth."})
    test_depth: List[int] = field(default_factory=lambda: [], metadata={'help': 'Specified evaluation depths.'})
    num_samples_per_case: int = field(default=1, metadata={'help': 'The number of samples in a case of \"length x depth\". Different samples differ in the contexts but their needles are the same.'})
    needle: str = field(default="\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n", metadata={'help': 'The needle content'})
    prompt: str = field(default='\n\nWhat is the best thing to do in San Francisco?\nAnswer:', metadata={'help': 'The needle content'})
    zh: bool = field(default=False, metadata={'help': 'Eval Chinese Text.'})
    num_syn_qa: int = field(default=0, metadata={'help': "The number of synthetic QA pairs."})
    syn_qa_needle_path: Optional[str] = field(default=None, metadata={'help': "The path to the prompts and the corresponding needles for TTT."})
    use_icl: bool = field(default=True)
    overwrite: bool = field(default=False, metadata={'help': "Overwrite the output file if it exists."})
    
    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
    
    def __post_init__(self):
        if len(self.test_length) == 0:
            self.test_length = np.linspace(self.test_length_min, self.test_length_max, self.test_length_num, endpoint=True).astype(int).tolist()
        elif self.test_length_min is not None or self.test_length_max is not None or self.test_length_num is not None:
            logger.warning("--test_length is provided so --test_length_min, --test_length_max, --test_length_num are ignored.")
        if len(self.test_depth) == 0:
            self.test_depth = np.linspace(self.test_depth_min, self.test_depth_max, self.test_depth_num, endpoint=True).astype(int).tolist()
        elif self.test_depth_min is not None or self.test_depth_max is not None or self.test_depth_num is not None:
            logger.warning("--test_length is provided so --test_length_min, --test_length_max, --test_length_num are ignored.")
        if self.num_syn_qa > 0 and self.syn_qa_needle_path is None:
            raise ValueError("--num_syn_qa > 0 but no --syn_qa_needle_path provided.")


class NeedleContextDataset(ICLContextDataset):
    def __init__(
        self,
        context: str,
        tokenizer: PreTrainedTokenizer,
        syn_qa_tasks: Optional[List],
        num_syn_qa: int,
        model_max_length: int,
        block_size: int,
        len_segment: int,
        len_offset: int,
        use_icl: bool=True
    ):
        """
        Args:
            context (`str`):
                The context.
            tokenizer (`transformers.PreTrainedTokenizer`):
                The tokenizer of the model to train.
            syn_qa_tasks (`List | None`):
                The pre-generated needle-similar auxiliary tasks.
            num_syn_qa (`int`):
                The number of auxiliary tasks.
            model_max_length (`int`):
                The size of the context window of the model to train. The texts will be clipped to fit in the context
                window.
            block_size (`int`):
                The number of tokens in a block; a block is the unit of segments and offsets.
            len_segment (`int`):
                The number of units in a segment; the article is divided into segments.
            len_offset (`int`):
                The number of units per offset; it determines the offset between one segment and the next one.
            use_icl (`bool`, *deprecated*):
                DEPRECATED! Fixed to True.
        """
        # insert synthetic needles into context
        if num_syn_qa > 0:
            sentences = sent_tokenize(context)
            syn_qa_tasks = np.random.choice(syn_qa_tasks, num_syn_qa, replace=False)
            for item in syn_qa_tasks:
                random_pos = np.random.randint(0, len(sentences))
                sentences.insert(random_pos, item['needle'])
            context = ' '.join(sentences)
        # generate the context dataset
        super().__init__(context, tokenizer, model_max_length, block_size, len_segment, len_offset)
        if num_syn_qa > 0:
            mixin = tokenizer("...", add_special_tokens=False)['input_ids']
            for item in syn_qa_tasks:
                if use_icl:
                    messages = [
                        {'role': 'system', 'content': "You are a helpful assistant."},
                        {'role': 'user', 'content': NIAHFORMAT_EN.format(context=context, prompt=item['prompt'])},
                        {'role': 'assistant', 'content': item['needle']}
                    ]
                else:
                    messages = [
                        {'role': 'system', 'content': "You are a helpful assistant."},
                        {'role': 'user', 'content': item['prompt']},
                        {'role': 'assistant', 'content': item['needle']}
                    ]
                input_length = len(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
                output_length = len(input_ids) - input_length
                if len(input_ids) > model_max_length:
                    input_ids = input_ids[:model_max_length//2-len(mixin)] + mixin + input_ids[-model_max_length//2:]
                    input_length = len(input_ids) - output_length
                self.data.append((input_ids, input_length))
        self.enable_qa_tag = False
    
    def enable_qa(self):
        self.enable_qa_tag = True
    
    def disable_qa(self):
        self.enable_qa_tag = False
    
    def __len__(self):
        return len(self.data) if self.enable_qa_tag else self.num_segments


def generate_sample(tokenizer: PreTrainedTokenizer, context: str, context_length: int, needle_depth: int, needle: str, prompt: str, zh: bool):
    if zh:
        num_words = len(context)
    else:
        num_words = len(context.split())
    if context_length > num_words:
        context = context * math.ceil(context_length / num_words)

    if zh:
        description = "以下上下文中隐藏着重要信息。找到并记住这些信息。我会问你关于其中重要信息的问题。\n"
    else:
        description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"

    description_input_ids = tokenizer.encode(description, add_special_tokens=False)
    needle_input_ids = tokenizer.encode(needle, add_special_tokens=False)
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    description_length = len(description_input_ids)
    needle_length = len(needle_input_ids)
    prompt_length = len(prompt_input_ids)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = context_length - prompt_length - needle_length - 1
    if minimum_pos > context_length or maximum_pos < 0:
        raise ValueError(f"The length {context_length} is too small. Please increase interval!")

    needle_pos = minimum_pos + round((maximum_pos - minimum_pos) * needle_depth / 100)
    
    context_input_ids = tokenizer.encode(context, max_length=context_length - description_length - needle_length - prompt_length, truncation=True, add_special_tokens=False)

    input_ids = sum([description_input_ids, context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:], prompt_input_ids], [])
    context_ids = sum([context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:]], [])
    inputs = tokenizer.decode(input_ids)
    context_return = tokenizer.decode(context_ids)

    return inputs, context_return, needle


def NIAH_Train(
    context: str,
    tokenizer: PreTrainedTokenizer,
    syn_qa_tasks: Optional[List],
    num_syn_qa: int,
    training_args: TrainingArguments,
    model_max_length: int,
    block_size: int,
    len_segment: int,
    len_offset: int,
    involve_qa_epochs: int,
    gather_batches: bool,
    model_name_or_path: str,
    use_lora: bool,
    lora_rank: Optional[int] = None,
    use_pissa: bool = False,
    load_in_4bit: bool = False,
    use_gated_memory: bool = False,
    use_icl: bool = True,
    **_
):
    context_dataset = NeedleContextDataset(context, tokenizer, syn_qa_tasks, num_syn_qa, model_max_length, block_size, len_segment, len_offset, use_icl=use_icl)
    model = load_model(
        model_name_or_path=model_name_or_path,
        use_lora=use_lora,
        lora_rank=lora_rank,
        use_pissa=use_pissa,
        load_in_4bit=load_in_4bit,
        vocab_size=len(tokenizer),
        use_gated_memory=use_gated_memory
    )
    model = train(
        model=model,
        dataset=context_dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        involve_qa_epochs=involve_qa_epochs,
        gather_batches=gather_batches
    )[0]
    return model


def generate_niah_input(niah_args: NIAHArgs, tokenizer: PreTrainedTokenizer):
    if os.path.isfile(niah_args.haystack_path):
        raise ValueError("Expected a directory but got a file.")
    elif os.path.isdir(niah_args.haystack_path):
        contexts = []
        haystack_files = [file for file in glob(f"{niah_args.haystack_path}/*.txt")]
        for i in range(niah_args.num_samples_per_case):
            num_tokens = 0
            contexts.append("")
            np.random.shuffle(haystack_files)
            for file in haystack_files:
                with open(file, 'r') as f:
                    this_file_context = f.read()
                    num_tokens += len(tokenizer(this_file_context, add_special_tokens=False))
                    contexts[i] += this_file_context
                    if num_tokens > max(niah_args.test_length):
                        break
    else:
        raise ValueError(f"Cannot find haystack: {niah_args.haystack_path}")
    all_inputs = []
    for length in tqdm(niah_args.test_length, desc="Constructing Data"):
        for depth in niah_args.test_depth:
            for i in range(niah_args.num_samples_per_case):
                prompt_case, context_case, needle_case = generate_sample(
                    tokenizer=tokenizer,
                    context=contexts[i],
                    context_length=length, 
                    needle_depth=depth,
                    needle=niah_args.needle,
                    prompt=niah_args.prompt,
                    zh=niah_args.zh
                )
                all_inputs.append(dict(
                    prompt=prompt_case,
                    context=context_case,
                    needle=needle_case,
                    length=length,
                    depth=depth
                ))
    if niah_args.cache_input_path is not None:
        with open(niah_args.cache_input_path, 'wb') as f:
            pickle.dump(all_inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cache the input file in {niah_args.cache_input_path}.")
    return all_inputs


def main():
    (niah_args, training_args, lift_args), config = parse_args((NIAHArgs, TrainingArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)), no_dict=(TrainingArguments, NIAHArgs), return_config=True)
    use_icl = niah_args.use_icl
    niah_args.needle = eval('\"\"\"' + niah_args.needle + '\"\"\"')
    niah_args.prompt = eval('\"\"\"' + niah_args.prompt + '\"\"\"')
    print(f"The prompt is:\n{niah_args.prompt}")
    print(f"The needle is:\n{niah_args.needle}")
    print(f"Test Lengths: {niah_args.test_length}")
    print(f"Test Depths: {niah_args.test_depth}")

    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    model_max_length = lift_args['model_max_length']
    output_path = niah_args.output_path
    cache_input_path = niah_args.cache_input_path
    num_resumed = 0
    if os.path.exists(output_path):
        if niah_args.overwrite:
            os.remove(output_path)
        else:
            num_resumed = len(open(output_path, 'r').readlines())
    
    # Load or create the input cache
    if cache_input_path is not None and os.path.exists(cache_input_path):
        logger.info(f"Load the cached input file {cache_input_path}.")
        with open(cache_input_path, "rb") as f:
            all_inputs = pickle.load(f)
    else:
        all_inputs = generate_niah_input(niah_args, tokenizer)
    # Load the needle tasks for LIFT
    if niah_args.syn_qa_needle_path is not None:
        with open(niah_args.syn_qa_needle_path, 'r') as f:
            lift_needle_tasks = json.load(f)
    else:
        lift_needle_tasks = None

    mixin = tokenizer("...", add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(1, -1)
    for sample in tqdm(all_inputs[num_resumed:], desc="Evaluating", initial=num_resumed, total=len(all_inputs)):
        prompt = sample['prompt']
        context = sample['context']
        model = NIAH_Train(
            context=context,
            tokenizer=tokenizer,
            syn_qa_tasks=lift_needle_tasks,
            num_syn_qa=niah_args.num_syn_qa,
            training_args=training_args,
            use_icl=use_icl,
            **lift_args
        )
        model.eval()
        with torch.no_grad():
            if use_icl:
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant."},
                    {'role': 'user', 'content': NIAHFORMAT_EN.format(context=context, prompt=prompt)}
                ]
            else:
                messages = [
                    {'role': 'system', 'content': "You are a helpful assistant."},
                    {'role': 'user', 'content': prompt}
                ]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
            if input_ids.shape[-1] > model_max_length:
                input_ids = torch.concat((input_ids[:, :model_max_length//2 - len(mixin)], mixin, input_ids[:, -model_max_length//2:]), dim=-1)
            input_ids = input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )[0]
        pred = tokenizer.decode(output_ids[input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"Prediction:\n{pred}")

        result = {
            'length': sample['length'],
            'depth': sample['depth'],
            'pred': pred,
            'needle': sample['needle']
        }
        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    main()
