import torch
import torch.utils
import torch.utils.data
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer,
    PreTrainedModel
)
from .context_dataset import ContextDataset
from typing import Optional, Tuple
from copy import deepcopy


def load_trainer(
    model: PreTrainedModel,
    training_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    training_args: TrainingArguments,
    eval_dataset: Optional[Dataset] = None,
    gather_batches: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[Trainer, PreTrainedModel]:
    """Load the HF trainer.
    Args:
        model (`transformers.PreTrainedModel`):
            The model to train.
        training_dataset (`torch.utils.data.Dataset`):
            The training dataset. In this program, it should be an instance of ContextDataset.
        tokenizer (`transformers.PreTrainedTokenizer`):
            The tokenizer of `model`.
        training_args (`transformers.TrainingArguments`):
            The HF training arguments.
        eval_dataset (`torch.utils.data.Dataset`, *optional*):
            The evaluation dataset. Defaults to None.
        gather_batches (`bool`, *optional*):
            Force the trainer to update the model only once every epoch. It is implemented with gradient accumulation
            and it may lead to more stable gradients. Defaults to False.
        optimizer (`torch.optim.Optimizer`, *optional*):
            The optimizer. If not provided, the trainer will initiate a new optimizer. Defaults to None.
    Returns:
        `tuple`:
            - trainer (`transformers.Trainer`): The trainer instance.
            - model (`transformers.PreTrainedModel`): The model to train.
    """
    training_args = deepcopy(training_args)
    if gather_batches:
        training_args.gradient_accumulation_steps = len(training_dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None)
    )
    return trainer, model


def train(
    model: PreTrainedModel,
    dataset: ContextDataset,
    tokenizer: PreTrainedTokenizer,
    training_args: TrainingArguments,
    involve_qa_epochs: int = 0,
    gather_batches: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """The LIFT process.
    Args:
        model (`PreTrainedModel`):
            The model to train.
        dataset (`ContextDataset`):
            The dataset for LIFT, containing the input segments (and auxiliary tasks).
        tokenizer (`transformers.PreTrainedTokenizer`):
            The tokenizer corresponding to the model to train.
        training_args (`transformers.TrainingArguments`):
            The HF training arguments.
        involve_qa_epochs (`int`, *optional*):
            The number of epochs of the 2nd stage. Defaults to 0.
        gather_batches (`bool`, *optional*):
            Force the trainer to update the model only once every epoch. It is implemented with gradient accumulation
            and it may lead to more stable gradients. Defaults to True.
    Returns:
        `tuple`:
            - model (`transformers.PreTrainedModel`): The LIFTed model.
            - tokenizer (`transformers.PreTrainedTokenizer`): The corresponding tokenizer.
    """
    # load tokenzier
    torch.cuda.empty_cache()  # Manually release memory
    # Load and finetune the model
    if involve_qa_epochs > 0:
        dataset.disable_qa()
    trainer, model = load_trainer(
        model=model,
        training_dataset=dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        gather_batches=gather_batches,
    )
    if training_args.num_train_epochs > 0:
        trainer.train()
    # Load the dataset with QA pairs and continue-finetune the model
    if involve_qa_epochs > 0:
        dataset.enable_qa()
        training_args_syn = deepcopy(training_args)
        training_args_syn.num_train_epochs = involve_qa_epochs
        trainer_syn, model = load_trainer(
            model=model,
            training_dataset=dataset,
            tokenizer=tokenizer,
            training_args=training_args_syn,
            gather_batches=gather_batches,
            optimizer=trainer.optimizer,
        )
        trainer_syn.train()
    # Clear cache
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
