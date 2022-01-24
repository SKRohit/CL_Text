import logging
import os
import sys
import pandas as pd
import torch
import pdb

from datasets import load_dataset
import data_utils

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
)
from arguments import ModelArguments, DataTrainingArguments


logger = logging.getLogger(__name__)

def inference_fucntion(model_args, data_args):

    if data_args.infer_file.endswith(".csv"):
        dataset = pd.read_csv(data_args.infer_file)
        column_names = dataset.columns
        if data_args.save_file_path is None:
            data_args.save_file_path = data_args.infer_file
    else:
        dataset = load_dataset(data_args.infer_file, split=data_args.split_name)
        column_names = dataset.column_names
    
    assert data_args.text_col in column_names, f"Please provide valid dataset. {data_args.text_col} is not present in {data_args.infer_file}."
    

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = model.eval()

    rev_model = AutoModelForSeq2SeqLM.from_pretrained(model_args.rev_model_name_or_path)
    rev_tokenizer = AutoTokenizer.from_pretrained(model_args.rev_model_name_or_path)
    rev_model = rev_model.eval()

    no_device = torch.cuda.device_count()
    if no_device>=2:
        device = torch.device("cuda:0")
        rev_device = torch.device("cuda:1")
    elif no_device==1:
        device = torch.device("cuda:0")
        rev_device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        rev_device = torch.device("cpu")
    
    model = model.to(device)
    rev_model = rev_model.to(rev_device)

    translated_text = []
    with torch.no_grad:
        for i in range(0,len(dataset),data_args.batch_size):
            inter_lang = []
            inputs = tokenizer(dataset[data_args.text_col][i:i+data_args.batch_size],
                        padding="longest",
                        max_length=256,
                        return_tensors="pt",).to(device)
            outputs = model.generate(inputs["input_ids"], max_length=data_args.max_length, num_beams=data_args.num_beams,)
            for output in outputs:
                inter_lang.append(tokenizer.decode(output,skip_special_tokens=True))
            
            inputs = rev_tokenizer(inter_lang, padding="longest",
                        max_length=256,return_tensors="pt").to(rev_device)
            outputs = rev_model.generate(inputs["input_ids"],max_length=data_args.max_length, num_beams=data_args.num_beams,)
            for output in outputs:
                translated_text.append(tokenizer.decode(output,skip_special_tokens=True))
            print(f"\rBatch {i+1} Finished.", flush=True)

    aug_cols = [col for col in column_names if col.startswith("aug_")]
    new_aug_col = "aug_"+str(len(aug_cols))

    if data_args.save_file_path.endswith(".csv"):
        if not data_args.save_only_new:
            dataset[new_aug_col] = translated_text
            dataset.to_csv(data_args.save_file_path,index=False)
        else:
            pd.DataFrame({data_args.text_col:dataset[data_args.text_col],
                            new_aug_col: translated_text},
                            ).to_csv(data_args.save_file_path,index=False)

    
if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    inference_fucntion(model_args, data_args)
