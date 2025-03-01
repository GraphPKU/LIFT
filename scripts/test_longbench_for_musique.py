"""
Test LIFT on the LongBench dataset (general cases).
Only 5 subtasks: gov_report, musique, narrativeqa, qmsum, passage_retrieval_en.
"""
from transformers import (
    TrainingArguments,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig
)
from lift.args import (
    ModelArguments,
    DataTrainingArguments,
    CustomTrainingArguments,
    parse_args
)
from lift.context_dataset import ContextDataset
from lift.model import load_tokenizer, load_model
from lift.train import train
from dataclasses import dataclass, field
from typing import List, Dict, Optional,Tuple
from numpy.random import randint
from nltk import sent_tokenize
import logging
import json
import os
import re
import torch
import tqdm
from torch.utils.data import Dataset
from copy import deepcopy


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

        front_len, back_len = get_fix_length_segments(len_lift_icl - 1, len_lift_icl - 1, len_lift_icl)
        front_st = randint(0, len_lift_icl - front_len)
        back_ed = randint(0, len_lift_icl - back_len)

        return input_ids[front_st: front_st+front_len], input_ids[-back_ed-back_len:-back_ed] if back_ed != 0 else input_ids[-back_len:]


SUBTASK_MAXLEN = {
    "narrativeqa": 128,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "passage_retrieval_en": 32,
}
SUBTASK_PROMPTS_ICL = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
}
SUBTASK_PROMPTS_NO_ICL = {
    'narrativeqa': "Based on your knowledge of the story, please answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nNow, answer the question based on the story you know as concisely as you can, using a single phrase if possible.\n\nQuestion: {input}\n\nAnswer:",
    'musique': "Answer the question based on the passages you know. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    'gov_report': "Based on your knowledge of the report by a government agency, write a one-page summary of the report.\n\nNow, write a one-page summary of the report you know.\n\nSummary:",
    'qmsum': "You are given a query containing a question or instruction. Based on the meeting transcript you know, answer the query in one or more sentences.\n\nNow, answer the query based on the meeting transcript you know in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "passage_retrieval_en": "You have known 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from. The following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from based on your knowledge of the 30 paragraphs. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
}
SYNFORMAT_NON_ICL = "Please answer the following question: {question}"
SYNFORMAT_ICL = "The article: \n{input}\nPlease answer the question based on the article.\nQuestion: {question}\nAnswer: "


@dataclass
class TestArguments:
    input_dir: str = field(default="", metadata={"help": "The LongBench dataset directory."})
    subtask_name: str = field(default="", metadata={"help": "The subtask name."})
    output_path: str = field(default="", metadata={"help": "The output path."})
    overwrite: bool = field(default=False, metadata={"help": "Overwrite the output file."})
    num_syn_qa: int = field(default=0, metadata={"help": "The number of synthetic QA pairs to generate; some subtasks are forced to use no synthetic QA pairs."})
    generator_name_or_path: Optional[str] = field(default=None, metadata={"help": "The generator model name or path."})
    use_icl: bool = field(default=True, metadata={"help": "Use ICL."})


class LongBenchDataset(ICLContextDataset):
    def __init__(
        self,
        context: str,
        tokenizer: PreTrainedTokenizer,
        model_max_length: int = 7800,
        block_size: int = 256,
        len_segment: int = 8,
        len_offset: int = 3,
        num_syn_qa: int = 0,
        generator_name_or_path: str = None,
        use_icl: bool = True
    ):
        """
        Args:
            context (`str`):
                The context (e.g. an article); the `context` attribute in LongBench.
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
            num_syn_qa (`int`, *optional*):
                The number of auxiliary tasks. Defaults to 0.
            generator_name_or_path (`str`, *optional*).
                The path to the generator used to synthesize auxiliary tasks. Required when `num_syn_qa > 0`. Defaults 
                to None.
            use_icl (`bool`, *deprecated*):
                DEPRECATED! Fixed to True.
        """
        super().__init__(context, tokenizer, model_max_length, block_size, len_segment, len_offset)
        # Generate QA pairs
        if num_syn_qa > 0:
            context_sent = sent_tokenize(context)
            assert len(context_sent) >= 16, "The length of the context should be at least 25 sentences."
            generator = AutoModelForCausalLM.from_pretrained(
                generator_name_or_path,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                ),
            )
            generator.eval()
            with tqdm.tqdm(total=num_syn_qa, desc="Generating synthetic QA pairs") as pbar:
                count_syn_qa = 0
                while count_syn_qa < num_syn_qa:
                    result = self.generate_task(generator, context, context_sent, model_max_length, use_icl)
                    if result is not None:
                        self.data.append(result)
                        count_syn_qa += 1
                        pbar.update(1)
            
        self.enable_qa_tag = False
    
    @torch.no_grad()
    def generate_task(self, generator: PreTrainedModel, full_context: str="", context_sent: List[str]=[], model_max_length: int=None, use_icl: bool=True):
        st_pos = randint(0, len(context_sent) - 16)
        context = ' '.join(context_sent[st_pos:st_pos+16])
        context = context.replace('\n', ' ')
        context = re.sub(r'\s+', ' ', context)
        messages = [
            {
                'role': "system",
                'content': "You are a helpful assistant."
            },
            {
                'role': "user", 
                'content': f"You are given a piece of text as the context. You should generate ONLY one question and the corresponding answer according to the context. You should also select one or more sentences directly from the original context as the evidence. The evidences must be EXACTLY SAME ADJACENT sentences retrieved from the context; KEEP the special tokens in the sentences. Please answer in the following format: \nQuestion: [question] \nAnswer: [answer] \nEvidence: [evidence]\nPlease DON'T output quotes when outputting evidences. The question should focus on the details like names, dates, e.t.c., and the answer should be as brief as possible. The following is the piece of text: {context}"
            }
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(generator.device)
        mask_attention = torch.ones_like(input_ids)
        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        for _ in range(3):
            outputs = generator.generate(
                input_ids=input_ids,
                attention_mask=mask_attention.to(generator.device),
                max_new_tokens=1024,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=True,
            )
            response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            question_position = response.find("Question:")
            answer_position = response.find("Answer:")
            evidence_position = response.find("Evidence:")
            if question_position == -1 or answer_position == -1 or evidence_position == -1:
                continue
            question = response[question_position + 9:answer_position].strip()
            answer = response[answer_position + 7:evidence_position].strip()
            evidence = response[evidence_position + 9:].strip()
            # if evidence not in context:
            #     continue
            break
        else:
            logging.warning("Fail to generate a QA pair, skip.")
            return None
        
        if use_icl:
            input_text = SYNFORMAT_ICL.format(input=full_context, question=question)
        else:
            input_text = SYNFORMAT_NON_ICL.format(question=question)
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': input_text},
            {'role': 'assistant', 'content': answer},
        ]
        
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        input_length = len(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)) 
        mixin = self.tokenizer("...", add_special_tokens=False)['input_ids']
        output_length = len(input_ids) - input_length
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            input_length = len(input_ids) - output_length
        
        return (input_ids, input_length)
    
    def enable_qa(self):
        self.enable_qa_tag = True
        
    def disable_qa(self):
        self.enable_qa_tag = False
    
    def __len__(self):
        return len(self.data) if self.enable_qa_tag else self.num_segments
    

def LongBenchtrain(context: str, tokenizer: PreTrainedTokenizer, model_name_or_path: str, training_args: TrainingArguments, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, use_lora: bool=False, lora_rank: Optional[int]=None, use_pissa: bool=False, load_in_4bit: bool=False, involve_qa_epochs: int=0, gather_batches: bool=True, num_syn_qa: int=0, use_gated_memory: bool=False, generator_name_or_path: Optional[str]=None, use_icl: bool=True, **kwargs):
    model = load_model(
        model_name_or_path=model_name_or_path,
        use_lora=use_lora,
        lora_rank=lora_rank,
        use_pissa=use_pissa,
        load_in_4bit=load_in_4bit,
        vocab_size=len(tokenizer),
        use_gated_memory=use_gated_memory
    )
    if use_lora or use_gated_memory:
        dataset = LongBenchDataset(context, tokenizer, model_max_length, block_size, len_segment, len_offset, num_syn_qa, generator_name_or_path, use_icl)
        model = train(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            training_args=training_args,
            involve_qa_epochs=involve_qa_epochs,
            gather_batches=gather_batches,
        )[0]
    return model


def prediction(data: List[Dict]=[], output_path: str="", num_syn_qa: int=0, training_args: TrainingArguments=None, lift_args: Dict=None, subtask_name: str=None, generator_name_or_path: Optional[str]=None, use_icl: bool=True):
    prompt_template = SUBTASK_PROMPTS_ICL[subtask_name] if use_icl else SUBTASK_PROMPTS_NO_ICL[subtask_name]
    max_new_tokens = SUBTASK_MAXLEN[subtask_name]
    model_max_length = lift_args['model_max_length']
    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    mixin = tokenizer("...", add_special_tokens=False, return_tensors='pt')['input_ids']
    for sample in tqdm.tqdm(data, desc="Predicting"):
        context, question = sample['context'], sample['input']
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': prompt_template.format(context=context, input=question)}
        ]
        model = LongBenchtrain(
            **lift_args,
            context=context,
            tokenizer=tokenizer,
            training_args=training_args,
            num_syn_qa=num_syn_qa,
            generator_name_or_path=generator_name_or_path,
            use_icl=use_icl,
        )
        model.eval()
        input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True)
        if input_ids.shape[-1] > model_max_length:
            input_ids = torch.concat([input_ids[:, :model_max_length//2-mixin.shape[-1]], mixin, input_ids[:, -model_max_length//2:]], dim=-1)
        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids)
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
        )
        output_ids = output_ids[:, input_ids.shape[-1]:]
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        sample['pred'] = pred
        with open(output_path, 'a') as f:
            f.write(json.dumps(sample) + '\n')
    return data

def main():
    training_args, test_args, lift_args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    input_dir = test_args.pop('input_dir')
    subtask_name = test_args.pop('subtask_name')
    if subtask_name != 'musique':
        raise ValueError("This program is designed for musique.")
    output_path = test_args.pop('output_path')
    overwrite = test_args.pop('overwrite')
    input_path = os.path.join(input_dir, f'{subtask_name}.jsonl')
    with open(input_path, 'r') as f:
        input_data = [json.loads(line) for line in f]
    num_resumed = 0
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            num_resumed = len(open(output_path, 'r').readlines())
    input_data = input_data[num_resumed:]
    prediction(
        data=input_data,
        output_path=output_path,
        training_args=training_args,
        lift_args=lift_args,
        subtask_name=subtask_name,
        **test_args
    )


if __name__ == '__main__':
    main()
