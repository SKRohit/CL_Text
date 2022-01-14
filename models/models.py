from dataclasses import dataclass
import torch
import torch.nn as nn

from transformers import AutoModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaConfig, RobertaEncoder
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertEncoder, BertConfig, load_tf_weights_in_bert

from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from typing import Optional, Tuple


@dataclass
class ContrastiveModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    """
    mlm_logits: Optional[torch.FloatTensor] = None
    pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SimCSEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    def __init__(self, config, *inputs, **kwargs):
        if "BertModel" in config.architectures:
            SimCSEPreTrainedModel.config_class = BertConfig
            SimCSEPreTrainedModel.base_model_prefix = "bert"
            SimCSEPreTrainedModel.load_tf_weights = load_tf_weights_in_bert
            SimCSEPreTrainedModel.supports_gradient_checkpointing = True
            SimCSEPreTrainedModel._keys_to_ignore_on_load_missing = [r"position_ids"]
        elif "RobertaModel" in config.architectures:
            SimCSEPreTrainedModel.config_class = RobertaConfig
            SimCSEPreTrainedModel.base_model_prefix = "roberta"
            SimCSEPreTrainedModel.supports_gradient_checkpointing = True
        super().__init__(config, *inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)        

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value
        elif isinstance(module, RobertaEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    
    # cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        if hasattr(cls, "model"):
            cls.mlp = cls.model.pooler
            cls.model.pooler = None
        cls.mlp = MLPLayer(config)
    cls.init_weights()


def cl_forward(cls,encoder,input_ids=None,attention_mask=None,token_type_ids=None,
        position_ids=None,head_mask=None,inputs_embeds=None,labels=None,
        output_attentions=None,output_hidden_states=None,sent_emb=False,
        return_dict=None,mlm_input_ids=None,
        ):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if not sent_emb:
        ori_input_ids = input_ids
        batch_size = input_ids.size(0)
        # Number of sentences in one instance. 2: instance with augmentation; 3: instance with augmentation and hard negative
        num_sent = input_ids.size(1)

        mlm_outputs = None
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

        if mlm_input_ids is not None:
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = encoder(
                    mlm_input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                    return_dict=True,
                )
            mlm_outputs = cls.lm_head(mlm_outputs.last_hidden_state)

    outputs = encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )

    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    if cls.model_args.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)
    

    if not return_dict:
        return (outputs[0], pooler_output) + (mlm_outputs,) + outputs[2:] 

    return ContrastiveModelOutput(
        mlm_logits=mlm_outputs,
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
        )

class PretrainedSimCSEForCL(SimCSEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model = AutoModel.from_config(config,)

        if self.model_args.do_mlm:
            if "BertModel" in self.config.architectures:
                self.lm_head = BertLMPredictionHead(config)
            elif "RobertaModel" in self.config.architectures:
                self.lm_head = RobertaLMHead(config)
            else:
                raise Exception("Please choose a pretrained SimCSE model shared by princeton-nlp.")

        cl_init(self, config)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,
        position_ids=None,head_mask=None,inputs_embeds=None,labels=None,
        output_attentions=None,output_hidden_states=None,return_dict=None,
        sent_emb=False,mlm_input_ids=None,
        ):

        return cl_forward(self,self.model,input_ids=input_ids,attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,head_mask=head_mask,
            inputs_embeds=inputs_embeds,labels=labels,output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,sent_emb=sent_emb,
            return_dict=return_dict,mlm_input_ids=mlm_input_ids,
            )

class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,
        position_ids=None,head_mask=None,inputs_embeds=None,labels=None,
        output_attentions=None,output_hidden_states=None,return_dict=None,
        sent_emb=False,mlm_input_ids=None,
        ):

        return cl_forward(self,self.bert,input_ids=input_ids,attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,head_mask=head_mask,
            inputs_embeds=inputs_embeds,labels=labels,output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,sent_emb=sent_emb,
            return_dict=return_dict,mlm_input_ids=mlm_input_ids,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,
        head_mask=None,inputs_embeds=None,labels=None,output_attentions=None,
        output_hidden_states=None,return_dict=None,sent_emb=False,mlm_input_ids=None,
        ):
        return cl_forward(self,self.roberta,input_ids=input_ids,attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,head_mask=head_mask,
            inputs_embeds=inputs_embeds,labels=labels,output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,sent_emb=sent_emb,
            return_dict=return_dict,mlm_input_ids=mlm_input_ids,
            )