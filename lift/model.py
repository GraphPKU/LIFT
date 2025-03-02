from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PeftConfig
)
from typing import Optional
from copy import deepcopy
import torch
from .gated_memory.model import GMLlamaForCausalLM


def load_tokenizer(tokenizer_name_or_path: str):
    """Load the tokenizer and set PAD_TOKEN = EOS_TOKEN.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name_or_path: str, load_in_4bit: bool=False, load_in_8bit: bool=False, vocab_size: Optional[int]=None, use_gated_memory: bool=False):
    """Load the base model.
    """
    if use_gated_memory:  # use_gated_memory should be checked first
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = GMLlamaForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config
            )
        elif load_in_8bit:
            raise NotImplementedError
        else:
            model = GMLlamaForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        for param in model.parameters():
            param.requires_grad_(False)
        model = PeftModel.from_pretrained(model, model_name_or_path, is_trainable=True)
        print(model)
        return model
    
    # We assume `peft` is available...
    from transformers.utils import find_adapter_config_file
    maybe_adapter_path = find_adapter_config_file(model_name_or_path)
    if maybe_adapter_path is not None:
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        model_base = load_base_model(peft_config.base_model_name_or_path, load_in_4bit, load_in_8bit, vocab_size)
        model = PeftModel.from_pretrained(model_base, model_name_or_path, config=peft_config, is_trainable=True)
        model = model.merge_and_unload()
        return model
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        model_base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
    elif load_in_8bit:
        raise NotImplementedError
    else:
        model_base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model_base.get_input_embeddings().weight.shape[0]
    if vocab_size is not None and vocab_size > embedding_size:
        model_base.resize_token_embeddings(vocab_size)
    model_base.enable_input_require_grads()
    return model_base


def load_model(
    model_name_or_path: str,
    use_lora: bool = False,
    lora_rank: Optional[int] = None,
    use_pissa: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    vocab_size: Optional[int] = None,
    use_gated_memory: bool = False
):
    """Load a trainable model.
    Args:
        model_name_or_path (`str`):
            The HF name or the local path to the model.
        use_lora (`bool`, *optional*):
            Use LoRA. Defaults to False.
        lora_rank (`int`, *optional*):
            The LoRA rank. Required if `use_lora=True`. Defaults to None.
        load_in_4bit (`bool`, *optional*):
            Use 4bit quantization. Defaults to False.
        load_in_8bit (`bool`, *deprecated*):
            Use 8bit quantization. NOT IMPLEMENTED! Defaults to False.
        vocab_size (`int`, *optional*):
            The vocabulary size. The model will adapt to the vocabulary size. Defaults to None.
        use_gated_memory (`bool`, *optional*):
            Use Gated Memory adapter. Defaults to False.
    Returns:
        model (`transformers.PreTrainedModel`): the model to train.
    """
    model_base = load_base_model(model_name_or_path, load_in_4bit, load_in_8bit, vocab_size, use_gated_memory)
    # Load the model
    if use_lora:
        # Init LoRA model
        if use_pissa:
            model = PeftModel.from_pretrained(model_base, model_name_or_path, subfolder="pissa_init", is_trainable=True)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                inference_mode=False,
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=.0,
                init_lora_weights='gaussian',
            )
            model = get_peft_model(model_base, peft_config)
    else:
        model = model_base
    torch.cuda.empty_cache()  # Manually release memory
    return model
