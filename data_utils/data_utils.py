import torch
from transformers import DataCollatorWithPadding




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
