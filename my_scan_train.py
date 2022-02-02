from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from functools import partial
import logging
import os
import pdb
import sys
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed,
    BertForPreTraining,
)
from transformers.trainer_utils import is_main_process

import data_utils
from models import SCANClustering
from arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments
from losses.losses import SCANLoss


MAX_GPU_BATCH_SIZE = 128
logger = logging.getLogger(__name__)

ddp_args = DistributedDataParallelKwargs()
ddp_args.find_unused_parameters = True

def training_fucntion(model_args, data_args, training_args):
    accelerator = Accelerator(fp16=True, cpu=False,kwargs_handlers=[ddp_args])
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    #Loading Tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,local_files_only=True, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,local_files_only=True, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")


    #Loading Tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name,local_files_only=True, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,local_files_only=True, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )


    #Loading Model
    if model_args.model_name_or_path:
        model = SCANClustering(
            config=config,
            model_args=model_args
        )
        # load pretrained-weights
        if model_args.pretrained_weight_path:
            state_dict = torch.load(model_args.pretrained_weight_path, map_location="cpu")

            if list(state_dict["model_state_dict"].keys())[0].startswith("module"):
                from collections import OrderedDict
                a = OrderedDict()
                for k,v in state_dict["model_state_dict"].items():
                    k = k[7:]
                    a[k]=v
                state_dict["model_state_dict"] = a
                
            model.backbone.load_state_dict(state_dict["model_state_dict"])
    else:
        raise Exception("Please provide model_name_or_path argument.")


    #Loading Dataset Containing Neighbours
    text_col = data_args.text_col
    # Prepare nbr dataset features
    if data_args.dataset_name:
        nbr_dataset = load_dataset(data_args.dataset_name)
    else:
        data_files = {k:v for k,v in zip(["train","validation","test"],[data_args.train_file,data_args.validation_file,data_args.test_file]) if v}
        nbr_dataset = load_dataset("csv", data_files = data_files)
    nbr_cnames = [cname for cname in nbr_dataset["train"].column_names if cname.startswith("nbr_")]
    prepare_scan_features = partial(data_utils.prepare_scan_features, tokenizer=tokenizer,text_col=text_col)
    for split in nbr_dataset:
        nbr_dataset[split] = nbr_dataset[split].map(prepare_scan_features, batched=True, num_proc=2)


    # Prepare aug dataset features
    aug_args = {"num_aug":None, "augment":False,}
    aug_dataset = {"train":None,"validation":None,"test":None}

    #Loading Augmented Dataset
    if data_args.use_augmentation:
        if data_args.dataset_name:
            aug_dataset = load_dataset(data_args.aug_dataset_name)
        else:
            data_files = {k:v for k,v in zip(["train","validation","test"],[data_args.aug_train_file,data_args.aug_validation_file,data_args.aug_test_file]) if v}
            aug_dataset = load_dataset("csv", data_files = data_files)

        aug_args["num_aug"] = len([cname for cname in aug_dataset["train"].column_names if cname.startswith("aug_")])
        aug_args["augment"] = True

        prepare_scan_features = partial(data_utils.prepare_scan_features, tokenizer=tokenizer,text_col=text_col,prefix="aug_")
        for split in aug_dataset:
            aug_dataset[split] = aug_dataset[split].map(prepare_scan_features, batched=True, num_proc=2)


    my_train_scan_collator = partial(data_utils.my_scan_collator, tokenizer=tokenizer,nbr_in_data=len(nbr_cnames), 
                                    num_nbr=data_args.num_nbr, sep_nbr=False,mlm=model_args.do_mlm,
                                    aug_dataset= aug_dataset["train"],**aug_args)
    my_validation_scan_collator = partial(data_utils.my_scan_collator, tokenizer=tokenizer,nbr_in_data=len(nbr_cnames), 
                                    num_nbr=data_args.num_nbr, sep_nbr=False,mlm=model_args.do_mlm,
                                    aug_dataset= aug_dataset["test"],**aug_args)

    train_dl = DataLoader(nbr_dataset["train"],batch_size=data_args.batch_size, collate_fn=my_train_scan_collator)
    val_dl = DataLoader(nbr_dataset["test"],batch_size=data_args.batch_size*2, collate_fn=my_validation_scan_collator)

    if data_args.batch_size > MAX_GPU_BATCH_SIZE:
        training_args.gradient_accumulation_steps = data_args.batch_size // MAX_GPU_BATCH_SIZE
        data_args.batch_size = MAX_GPU_BATCH_SIZE


    model = model.to(accelerator.device)
    optimizer = AdamW(params=model.parameters(), lr=training_args.learning_rate, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                    num_warmup_steps=100, 
                    num_training_steps=(training_args.num_train_epochs * len(train_dl)//training_args.gradient_accumulation_steps),
                    )
    loss_func = SCANLoss(entropy_weight=model_args.entropy_weight,)
    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)
    optimizer.zero_grad()

    for epoch in range(int(training_args.num_train_epochs)):
        
        accelerator.print(f"Training Epoch {epoch+1} Started.")
        total_train_loss = 0

        for step, batch in enumerate(train_dl):
            if training_args.keep_eval_mode:
                model.eval()
            else:
                model.train()
            # model.train()
            batch = {k:v.to(accelerator.device) for k,v in batch.items()}
            
            if training_args.update_cluster_head_only:
                with torch.no_grad():
                    outputs = model(batch, forward_pass="backbone")
                outputs = model(outputs.pooler_output,forward_pass="head")
            else:
                outputs = model(batch)
            
            # Loss for every head
            total_loss = calculate_loss(loss_func, outputs)
            total_loss = torch.sum(torch.stack(total_loss, dim=0))
            total_loss = total_loss / training_args.gradient_accumulation_steps

            total_train_loss += total_loss.item()
            accelerator.backward(total_loss)
            if step % training_args.gradient_accumulation_steps == 0:
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
                        outputs = model(batch)
                    # Loss for every head
                    total_loss_eval = calculate_loss(loss_func, outputs)

                    total_loss_eval = torch.min(torch.stack(total_loss_eval, dim=0))  # considering lowest loss out of the losses on different heads
                    total_eval_loss += total_loss_eval.item()
                
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

def calculate_loss(loss_func, outputs):
    total_loss = []
    for output in outputs:
        length = output.size()[0]
        total_loss_, consistency_loss_, entropy_loss_ = loss_func(output[:length//2],output[length//2:])
        total_loss.append(total_loss_)
    return total_loss

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
