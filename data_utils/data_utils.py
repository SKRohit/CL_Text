from random import randint
import torch
from transformers import DataCollatorWithPadding


def prepare_scan_features(examples, text_col, tokenizer, prefix="nbr_"):
    nbr_cnames = [k for k in examples if k.startswith(prefix)]
    total = len(examples[text_col])

    for idx in range(total):
        if examples[text_col][idx] is None:
            examples[text_col][idx] = " "
        for cname in nbr_cnames:
            if examples[cname][idx] is None:
                examples[cname][idx] = " "
    
    sentences = examples[text_col] + [examples[cname][idx] for cname in nbr_cnames for idx in range(total)]
    sent_features = tokenizer(sentences,padding=True,truncation=True, max_length=256,)
    features = {}
    for key in sent_features:
        temp = sent_features[key]
        features[text_col+'_'+key] = sent_features[key][:total]
        i = total
        for cname in nbr_cnames:
            features[cname+'_'+key] = temp[i:i+total]
            i += total
    return features


def my_scan_collator(features, tokenizer, nbr_in_data, num_nbr=None, sep_nbr=False, num_aug=None, augment=True, aug_dataset=None, mlm=False):
    special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
    
    if num_nbr is not None:
        assert nbr_in_data >= num_nbr
        num_nbr = randint(1,num_nbr)
    else:
        num_nbr = randint(1,nbr_in_data)

    flat_features = {k:[] for k in special_keys}
    for x in ["text", "nbr_"+str(num_nbr)]:
        for k in special_keys:
            for feature in features:
                if feature.get(x+"_"+k):
                    if augment and aug_dataset:
                        aug_data = aug_dataset[feature['index']]
                        assert aug_data['index']==feature['index'], "Mismatch in augmentation"
                        cname = "aug_"+str(randint(1,num_aug))+"_"+k
                        flat_features[k].append(aug_data[cname])
                    else:
                        flat_features[k].append(feature[x+"_"+k])

    flat_features = {k:v for k,v in flat_features.items() if v}

    batch = DataCollatorWithPadding(tokenizer, padding=True, return_tensors="pt")(flat_features)

    if mlm:            
        batch["mlm_input_ids"], batch["mlm_labels"] = mask_tokens(batch["input_ids"],tokenizer=tokenizer)

    if sep_nbr:
        batch = {"actual_text":{k:v[:len(v)//2] for k,v in batch.items()},
                 "nbr_text":{k:v[len(v)//2:] for k,v in batch.items()}}

    return batch


def prepare_cl_features(examples, sent0_cname, sent1_cname, tokenizer):
    total = len(examples[sent0_cname])

    # Avoid "None" fields 
    for idx in range(total):
        if examples[sent0_cname][idx] is None:
            examples[sent0_cname][idx] = " "
        if examples[sent1_cname][idx] is None:
            examples[sent1_cname][idx] = " "

    sentences = examples[sent0_cname] + examples[sent1_cname]
    import pdb
    # pdb.set_trace()
    # import pandas as pd
    # sentences = pd.DataFrame(sentences).dropna().values.tolist()
    
    sent_features = tokenizer(sentences,max_length=256,truncation=True)
    
    features = {}
    for key in sent_features:
        features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]

    return features


def my_cl_collator(features,tokenizer,mlm=False):
    special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
    bs = len(features)
    if bs > 0:
        num_sent = len(features[0]['input_ids'])
    else:
        return
    flat_features = []
    for feature in features:
        for i in range(num_sent):
            flat_features.append({k: feature[k][i] for k in feature if k in special_keys})

    batch = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")(flat_features)
    if mlm:
        batch["mlm_input_ids"], batch["mlm_labels"] = mask_tokens(batch["input_ids"],tokenizer=tokenizer)
    batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

    if "label" in batch:
        batch["labels"] = batch["label"]
        del batch["label"]
    if "label_ids" in batch:
        batch["labels"] = batch["label_ids"]
        del batch["label_ids"]

    return batch


def mask_tokens(inputs = None, special_tokens_mask=None, tokenizer=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    inputs = inputs.clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, 0.15)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
