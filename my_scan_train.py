import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import pdb
import random
from functools import partial
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType

from datasets import load_dataset
import data_utils

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed,
    BertForPreTraining,
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from models import PretrainedSimCSEForCL, BertForCL, RobertaForCL, SCANClustering
from arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from losses.losses import SimCSELoss, SCANLoss


MAX_GPU_BATCH_SIZE = 128
logger = logging.getLogger(__name__)

def training_fucntion(model_args, data_args, training_args):
    accelerator = Accelerator(fp16=True, cpu=False)
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    gradient_accumulation_steps=4
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    datasets = load_dataset(data_args.train_file,)# cache_dir="./data/",)# delimiter="\t" if "tsv" in data_args.train_file else ",")
    
    
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if 'san' in model_args.model_name_or_path:
            model = SCANClustering(
                config=config,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.backbone.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())

        if 'simcse' in model_args.model_name_or_path:
            model = PretrainedSimCSEForCL(
                config=config,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())

        elif 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        
        
        else:
            raise NotImplementedError
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    # model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    prepare_cl_features = partial(data_utils.prepare_cl_features, tokenizer=tokenizer, sent0_cname=sent0_cname, sent1_cname=sent1_cname,)
    my_cl_collator = partial(data_utils.my_cl_collator, tokenizer=tokenizer,mlm=model_args.do_mlm)
    
    train_dataset = datasets["train"].map(prepare_cl_features,batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    val_dataset = datasets["test"].map(prepare_cl_features,batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    train_dl = DataLoader(train_dataset,batch_size=data_args.batch_size, collate_fn=my_cl_collator)
    val_dl = DataLoader(val_dataset,batch_size=data_args.batch_size*2, collate_fn=my_cl_collator)

    if data_args.batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = data_args.batch_size // MAX_GPU_BATCH_SIZE
        data_args.batch_size = MAX_GPU_BATCH_SIZE

    model = model.to(accelerator.device)
    optimizer = AdamW(params=model.parameters(), lr=training_args.learning_rate, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                    num_warmup_steps=100, 
                    num_training_steps=(training_args.num_train_epochs * len(train_dl)//gradient_accumulation_steps),
                    )
    loss_func = SimCSELoss(temp=model_args.temp, device=accelerator.device)

    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    for epoch in range(int(training_args.num_train_epochs)):
        accelerator.print(f"Training Epoch {epoch+1} Started.")
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dl):
            batch = {k:v.to(accelerator.device) for k,v in batch.items()}
            outputs = model(**batch)
            if model_args.do_mlm:
                loss = loss_func(outputs.pooler_outut, outputs.mlm_logits,batch["mlm_labels"])
            else:
                loss = loss_func(outputs.pooler_output)
            total_train_loss += loss.item()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.print(f"\rUpdating Weights after {step+1}/{len(train_dl)} steps.", end='', flush=True)
                accelerator.clip_grad_norm_(model.parameters(),training_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            torch.cuda.empty_cache()
        ######## EVALUATIOn ON VALIDATION SET ###########
            if (step+1)%training_args.eval_steps==0 or (step+1)==len(train_dl):
                model.eval()
                total_eval_loss=0
                for step, batch in enumerate(val_dl):
                    batch = {k:v.to(accelerator.device) for k,v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    if model_args.do_mlm:
                        loss = loss_func(outputs.pooler_outut, outputs.mlm_logits,batch["mlm_labels"])
                    else:
                        loss = loss_func(outputs.pooler_output)
                    total_eval_loss += loss.item()
                
                accelerator.print(f"epoch {epoch+1}: train_loss: {total_train_loss/len(train_dl)}, val_loss: {total_eval_loss/len(val_dl)},")
                total_train_loss = 0

                # save the model if metric on validation set increases
                if total_eval_loss < data_args.best_eval_metric:   
                    accelerator.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, f"{training_args.output_dir}/{epoch+1}_{total_eval_loss}.checkpoint.pth.tar")
                    data_args.best_eval_metric = total_eval_loss
                    accelerator.print("\n\n\n Best Model is being Saved \n\n\n")

if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    training_fucntion(model_args, data_args, training_args)
