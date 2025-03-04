"""
Test LIFT on the LooGLE dataset (general cases).
For timeline reorder tasks only please refer to test_loogle_timeline.py.
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
import torch
import tqdm
from torch.utils.data import Dataset
from copy import deepcopy


LOOGLEFORMAT_NON_ICL = "Based on the article {title}, please answer the following question: {question}"
LOOGLEFORMAT = "The article {title}: \n{input}\nPlease answer the question based on {title}.\nQuestion: {question}\nAnswer: "
LOOGLEFORMAT_COT = """The article {title}:
{input}
Please recall one or several original sentences from article {title} as evidence, and then answer the question solely based on this evidence.
Please answer in the following format:

# Evidence:
[evidence]
# Answer:
[answer]
# End of answer

Please DON'T output quotes when outputing evidences.
# Question:
{question}
# Evidence:
"""


@dataclass
class TestArguments:
    input_file: str = field(metadata={"help": "The input file for the test."})
    output_file: str = field(metadata={"help": "The output file for the test."})
    overwrite: bool = field(default=False, metadata={"help": "Whether to overwrite the output file."})
    num_syn_qa: int = field(default=0, metadata={"help": "The number of synthetic QA pairs to generate."})
    title_option: int = field(default=1, metadata={"help": "The title option for the LooGLE dataset."})
    generator_name_or_path: Optional[str] = field(default=None, metadata={"help": "The generator model name or path."})
    use_cot: bool = field(default=False, metadata={'help': "Whether to use CoT in syn. QA and test."})
    num_test: Optional[int] = field(default=None, metadata={'help': "Test only the first several articles."})
    use_icl: bool = field(default=True, metadata={'help': "Whether to use ICL when training and testing."})


class LooGLEDataset(ContextDataset):
    def __init__(self, context: str, title: str, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False, use_icl: bool=True):
        # Option 1: prepend title before context
        if title_option == 1:
            context = title + '\n' + context
        super().__init__(context, tokenizer, model_max_length, block_size, len_segment, len_offset)
        # Option 2: prepend title before each segment and predict the whole segment
        if title_option == 2:
            snippet = tokenizer(f"A snippet of {title}: ", add_special_tokens=False)['input_ids']
            self.data = [(snippet + input_ids, 0) for input_ids, _ in self.data]
        # Option 3: prepend title before each segment and predict the content
        if title_option == 3:
            snippet = tokenizer(f"A snippet of {title}: ", add_special_tokens=False)['input_ids']
            self.data = [(snippet + input_ids, len(snippet)) for input_ids, _ in self.data]
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
                    do_sample=False,
                ),
            )
            gen_tokenizer = load_tokenizer(generator_name_or_path)
            generator.eval()
            for _ in range(num_syn_qa):
                result = self.generate_task(generator, gen_tokenizer, context, context_sent, title, model_max_length, use_cot, use_icl=use_icl)
                if result is not None:
                    self.data.append(result)
        self.enable_qa_tag = False
    
    @torch.no_grad()
    def generate_task(self, generator: PreTrainedModel, tokenizer: PreTrainedTokenizer, full_context: str, context_sent: List[str], title: str, model_max_length: int, use_cot: bool=False, use_icl: bool=True):
        st_pos = randint(0, len(context_sent) - 16)
        context = ' '.join(context_sent[st_pos:st_pos+16])
        messages = [
            {
                'role': "system",
                'content': "You are a helpful assistant."
            },
            {
                'role': "user", 
                'content': f"You are given a piece of text as the context. You should generate ONLY one question and the corresponding answer according to the context. You should also select one or more sentences directly from the original context as the evidence. The evidences must be verbatim sentences from the context. Please answer in the following format: \nQuestion: [question] \nAnswer: [answer] \nEvidence: [evidence]\nPlease DON'T output quotes when outputting evidences. The following is the piece of text: {context}"
            }
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(generator.device)
        mask_attention = torch.ones_like(input_ids)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        for _ in range(5):
            outputs = generator.generate(
                input_ids=input_ids,
                attention_mask=mask_attention.to(generator.device),
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=False,
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            question_position = response.find("Question:")
            answer_position = response.find("Answer:")
            evidence_position = response.find("Evidence:")
            if question_position == -1 or answer_position == -1 or evidence_position == -1:
                continue
            question = response[question_position + 9:answer_position].strip()
            answer = response[answer_position + 7:evidence_position].strip()
            evidence = response[evidence_position + 9:].strip()
            if evidence not in context:
                continue
            break
        else:
            logging.warning("Fail to generate a QA pair, skip.")
            return None
        
        if not use_icl:
            input_text = LOOGLEFORMAT_NON_ICL.format(title=title, question=question)
        else:
            if use_cot:
                input_text = LOOGLEFORMAT_COT.format(title=title, input=full_context, question=question)
                answer = evidence + "\n# Answer:\n" + answer + "\n# End of answer"
            else:
                input_text = LOOGLEFORMAT.format(title=title, input=full_context, question=question)

        # print(f'syn input text: {input_text}')
        example = input_text + ' ' + answer
        input_ids = self.tokenizer(example, add_special_tokens=False)['input_ids']
        input_length = len(self.tokenizer(input_text, add_special_tokens=False)['input_ids']) 
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
    
    
def LooGLEtrain(context: str, title: str, tokenizer: PreTrainedTokenizer, model_name_or_path: str, training_args: TrainingArguments, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, use_lora: bool=False, lora_rank: Optional[int]=None, use_pissa: bool=False, load_in_4bit: bool=False, involve_qa_epochs: int=0, gather_batches: bool=True, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_gated_memory: bool=False, use_cot: bool=False, use_icl: bool=True, **kwargs):
    model = load_model(model_name_or_path=model_name_or_path, use_lora=use_lora, lora_rank=lora_rank, use_pissa=use_pissa, load_in_4bit=load_in_4bit, vocab_size=len(tokenizer), use_gated_memory=use_gated_memory)
    if use_lora or use_gated_memory:
        dataset = LooGLEDataset(context, title, tokenizer, model_max_length, block_size, len_segment, len_offset, num_syn_qa, title_option, generator_name_or_path, use_cot, use_icl=use_icl)
        model = train(model, dataset, tokenizer, training_args, involve_qa_epochs, gather_batches)[0]
    return model


def prediction(data: List[Dict], training_args: TrainingArguments, lift_args: Dict, output_file: str, num_resumed: int=0, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False, use_icl: bool=True):
    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    mixin = tokenizer("...", add_special_tokens=False)['input_ids']
    model_max_length = lift_args['model_max_length']
    
    for i, sample in enumerate(tqdm.tqdm(data, desc="Sample")):
        if i < num_resumed:
            continue
        context = sample['input']
        title = sample['title']
        # qa_pairs = sample['test_qa_pairs']
        qa_pairs = eval(sample['qa_pairs'])
        model = LooGLEtrain(context, title, tokenizer, training_args=training_args, num_syn_qa=num_syn_qa, title_option=title_option, generator_name_or_path=generator_name_or_path, use_cot=use_cot, use_icl=use_icl, **lift_args)
        model.eval()
        for qa_pair in tqdm.tqdm(qa_pairs, desc="QA Pair"):

            if not use_icl:
                input_text = LOOGLEFORMAT_NON_ICL.format(title=title, question=qa_pair['Q'])
            else:
                if use_cot:
                    input_text = LOOGLEFORMAT_COT.format(title=title, input=context, question=qa_pair['Q'])
                else:
                    input_text = LOOGLEFORMAT.format(title=title, input=context, question=qa_pair['Q'])

            # print(f'input_text: {input_text}')
            input_ids = tokenizer(input_text, add_special_tokens=False)['input_ids']
            if len(input_ids) > model_max_length:
                input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            # terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=terminators,
                max_new_tokens=200,
                use_cache=True,
                do_sample=False,
            )
            response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            qa_pair['pred'] = response
        output_case = {
            'title': title,
            'input': context,
            'qa_pairs': qa_pairs
        }
        with open(output_file, 'a') as f:
            f.write(json.dumps(output_case) + '\n')


def main():
    training_args, test_args, lift_args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    input_file = test_args.pop('input_file')
    output_file = test_args.pop('output_file')
    overwrite = test_args.pop('overwrite')
    num_test = test_args.pop('num_test')
    use_icl = test_args.pop('use_icl')

    if not use_icl:
        assert test_args['use_cot'] is False, "CoT is not supported in non-ICL mode."

    num_resumed = 0
    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            with open(output_file, 'r') as f:
                num_resumed = len(f.readlines())
    with open(input_file, 'r') as f:
        input_data = [json.loads(line) for line in f]
    if num_test is not None:
        input_data = input_data[:num_test]
    prediction(input_data, training_args, lift_args, output_file, num_resumed=num_resumed, use_icl=use_icl, **test_args)


if __name__ == '__main__':
    main()
