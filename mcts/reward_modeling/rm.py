import os
import re
import copy
import time
import torch
import random
import requests
import nltk
# import sglang as sgl
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter, defaultdict
from models import OpenAIModel
from nltk import sent_tokenize
from vllm import LLM, SamplingParams
# from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import numpy as np
from utils.utils import load_prompt_template, remove_citations
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import (
    InputExample,
    InputFeatures,
    GPT2LMHeadModel,
    BertTokenizer,
    BertForSequenceClassification,
)
from tetra_exp.ICL_selection.bart_score import BARTScorer

class ModelArgsWrapper():
    def __init__(self, args=None):
        super(ModelArgsWrapper, self).__init__()
        if args is not None:
            for k, v in args.items():
                setattr(self, k, v)


    def assign_properties(self, d):
        for k, v in d.items():
            setattr(self, k, v)     
class SOME(nn.Module):
    def __init__(self, args_dict):
        super(SOME, self).__init__()

        device_str = 'cpu'
        if torch.cuda.is_available():
            device_str = 'cuda:{}'.format(2)

        self.device = torch.device(device_str)
        self.args = ModelArgsWrapper(args_dict)
        self.model_g = BertForSequenceClassification.from_pretrained(self.args.g_dir)
        self.model_f = BertForSequenceClassification.from_pretrained(self.args.f_dir)
        self.model_m = BertForSequenceClassification.from_pretrained(self.args.m_dir)
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_type)

        
    def convert_examples_to_features(
        self,
        examples,
        tokenizer,
        max_length=None,
        task=None,
        label_list=None,
        output_mode=None,
    ):
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample):
            if example.label is None:
                return None
            elif output_mode == 'classification':
                return label_map[example.label]
            elif output_mode == 'regression':
                return float(example.label)
            raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        return features


    def create_example(self, src, pred, task):
        examples = []
        if task == 'ssreg':
            for i, (s, p) in enumerate(zip(src, pred)):
                examples.append(
                    InputExample(guid=i, text_a=s, text_b=p, label=None)
                )
        elif task == 'sreg':
            for i, p in enumerate(pred):
                examples.append(
                    InputExample(guid=i, text_a=p, text_b=None, label=None)
                )
        return examples


    def create_dataset(self, src, pred, task=None):
        # load examples and convert to features
        examples = self.create_example(src, pred, task=task)
        tokenizer = self.tokenizer
        features = self.convert_examples_to_features(
            examples,
            tokenizer,
            label_list=[None],
            max_length=128,
            output_mode='regression',
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        data = {
            'input_ids': all_input_ids.to(self.device),
            'attention_mask': all_attention_mask.to(self.device),
        }

        if self.args.model_type == 'distilbert' or \
            ('bert' not in self.args.model_type and 'xlnet' not in self.args.model_type):
            all_token_type_ids = None
        else:
            all_token_type_ids = all_token_type_ids.to(self.device)
        data['token_type_ids'] = all_token_type_ids

        return data


    def predict(self, task):
        if task == 'grammer':
            some_model = self.model_g
            pred_dataset = self.data_sreg
        elif task == 'fluency':
            some_model = self.model_f
            pred_dataset = self.data_sreg
        elif task == 'meaning':
            some_model = self.model_m
            pred_dataset = self.data_ssreg

        some_model.to(self.device)

        preds = None
        some_model.eval()

        with torch.no_grad():                
            outputs = some_model(**pred_dataset)
            logits = outputs[:2][0]

        preds = logits.detach().cpu().numpy()
        preds = np.squeeze(preds, axis=-1)

        return preds


    def add(self, src, pred):
        if not isinstance(src, list):
            src = [src]
        if len(src) == 1:
            src = src * len(pred)
        # make dataset for sreg and ssreg
        self.data_sreg = self.create_dataset(src, pred, task='sreg')
        self.data_ssreg = self.create_dataset(src, pred, task='ssreg')


    def min_max_normalize(self, x, x_min=1, x_max=4):
        return (x - x_min) / (x_max - x_min)

    
    def score(self, sources, hyps):
        self.add(sources, hyps)

        # normalize
        score_g = [self.min_max_normalize(x) for x in self.predict(task='grammer')]
        score_f = [self.min_max_normalize(x) for x in self.predict(task='fluency')]
        score_m = [self.min_max_normalize(x) for x in self.predict(task='meaning')]
        
        # assert len(score_g) == len(score_f) == len(score_m)

        # calc gfm score
        scores = []
        for g, f, m in zip(score_g, score_f, score_m):
            scores.append(
                self.args.weight_g * g + self.args.weight_f * f + self.args.weight_m * m
            )

        return score_g, score_f, score_m
    

class RM:
    def __init__(self):
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.gpt2_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        self.gpt2_model.to(self.gpt2_device)

        self.some_model_args = {'model_type': 'bert-base-cased',
                                'g_dir': os.path.join(''),
                                'f_dir': os.path.join(''),
                                'm_dir': os.path.join(''),
                                'weight_g': 0.55,
                                'weight_f': 0.43,
                                'weight_m': 0.02}
        self.some_model = SOME(self.some_model_args)
        self.bart_scorer = BARTScorer(device='cuda:3', checkpoint='facebook/bart-large-cnn')

    #### need to define the list of quality checking functions here
    def word_count_check(self, count, relation, polished_paragraph):
        tot_word_count = len(nltk.word_tokenize(polished_paragraph))
        int_count = int(count)
        label = 0
        feedback = "The word count for the polished text is {}, it follows the user's requirement ({} {} words).".format(tot_word_count, relation, int_count)
        if relation == 'less than':
            if tot_word_count < int_count:
                label = 1
            else:
                feedback = "The word count for the polished text should be less than {}, your word count is {}, please consider reducing it.".format(int_count, tot_word_count)
        elif relation == 'more than':
            if tot_word_count > int_count:
                label = 1
            else:
                feedback = "The word count for the polished text should be less than {}, your word count is {}, please consider increasing it.".format(int_count, tot_word_count)
        else:
            if tot_word_count == int_count:
                label = 1
            elif tot_word_count < int_count:
                feedback = "The word count for the polished text should be {}, your word count is {}, please consider increasing it.".format(int_count, tot_word_count)
            else:
                feedback = "The word count for the polished text should be {}, your word count is {}, please consider reducing it.".format(int_count, tot_word_count)
        # print(tot_word_count)
        # print(relation)
        # print(label)
        return tot_word_count, label, feedback

    def keyword_frequency_check(self, keyword, frequency, relation, polished_paragraph):
        label = 0
        pol_text_lower = polished_paragraph.lower()
        keyword_lower = keyword.lower()
        cur_cnt = pol_text_lower.count(keyword_lower)
        int_f = int(frequency)
        feedback = "The keyword \"{}\" occured {} times, it follows the user's requirement ({} {} times).".format(keyword, cur_cnt, relation, int_f)
        if relation == "less than":
            if cur_cnt < int_f:
                label = 1
            else:
                feedback = "The keyword \"{}\" should occur less than {} times, its occurence in your text is {} times, please consider reducing it.".format(keyword, int_f, cur_cnt)
        elif relation == "more than":
            if cur_cnt > int_f:
                label = 1
            else:
                feedback = "The keyword \"{}\" should occur more than {} times, its occurence in your text is {} times, please consider increasing it.".format(keyword, int_f, cur_cnt)
        else:
            if cur_cnt == int_f:
                label = 1
            elif cur_cnt < int_f:
                feedback = "The keyword \"{}\" should occur {} times, its occurence in your text is {} times, please consider increasing it.".format(keyword, int_f, cur_cnt)
            else:
                feedback = "The keyword \"{}\" should occur {} times, its occurence in your text is {} times, please consider reducing it.".format(keyword, int_f, cur_cnt)
        return cur_cnt, label, feedback

    def keyword_keep_removal_check(self, keyword, relation, polished_paragraph, ori_text):
        label = 0
        feedback = ""
        if relation == "keep":
            ori_text_lower = ori_text.lower()
            pol_text_lower = polished_paragraph.lower()
            keyword_lower = keyword.lower()
            ori_cnt = ori_text_lower.count(keyword_lower)
            cur_cnt = pol_text_lower.count(keyword_lower)
            if ori_cnt == cur_cnt:
                label = 1
                feedback = "The keyword \"{}\" has been kept, it follows the user's requirement.".format(keyword)
            else:
                feedback = "The keyword \"{}\" has been removed from the sentence, please conside adding them back.".format(keyword)
        else: ### "remove"
            if keyword not in polished_paragraph:
                label = 1
                feedback = "The keyword \"{}\" has been removed, it follows the user's requirement.".format(keyword)
            else:
                feedback = "The keyword \"{}\" exist in the text, please conside removing it.".format(keyword)
        return label, feedback

    def split_into_sentences(self, paragraph):
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        sentences = [re.sub(r'[.!?]$', '', sentence) for sentence in sentences]
        return sentences

    def sentence_modification_check(self, sentence_id, relation, polished_paragraph, ori_text):
        ori_sent_list = self.split_into_sentences(ori_text)
        sent_list = []
        if relation == "change":
            if type(sentence_id) == int:
                sentence_id = [sentence_id]
            for sent_id in sentence_id:
                ori_sent = ori_sent_list[sent_id - 1]
                if ori_sent in polished_paragraph:
                    feedback = "The sentence '{}' has not been changed, please consider modifying it.".format(ori_sent)
                    return 0, feedback
                sent_list.append(ori_sent)
            sentences = "; ".join(sent_list)
            feedback = "The targeted sentence has been changed, it follows the user's requirement. Please keep them unchanged: {}".format(sentences)
            return 1, feedback
        else:
            if type(sentence_id) == int:
                sentence_id = [sentence_id]
            for sent_id in sentence_id:
                ori_sent = ori_sent_list[sent_id - 1]
                if ori_sent not in polished_paragraph:
                    feedback = "The sentence '{}' has been changed, please do not modify this sentence.".format(ori_sent)
                    return 0, feedback
                sent_list.append(ori_sent)
            sentences = "; ".join(sent_list)
            feedback = "The targeted sentence has not been changed, it follows the user's requirement. Please keep them unchanged: {}".format(sentences)
            return 1, feedback

    def sentence_count_check(self, count, relation, polished_paragraph):
        label = 0
        polished_sent_list = self.split_into_sentences(polished_paragraph)
        sent_cnt = len(polished_sent_list)
        int_c = int(count)
        feedback = "The polished text contains {} sentences, it follows the user's requirement ({} {} sentences).".format(sent_cnt, relation, int_c)
        if relation == "less than":
            if sent_cnt < int_c:
                label = 1
            else:
                feedback = "The polished text contains {} sentences. However, it should contain less than {} sentences, please consider reducing the number of sentences in the polished text.".format(sent_cnt, int_c)
        elif relation == "more than":
            if sent_cnt > int_c:
                label = 1
            else:
                feedback = "The polished text contains {} sentences. However, it should contain more than {} sentences, please consider increasing the number of sentences in the polished text.".format(sent_cnt, int_c)
        else:
            if sent_cnt == int_c:
                label = 1
            elif sent_cnt < int_c:
                feedback = "The polished text contains {} sentences. However, it should contain {} sentences, please consider increasing the number of sentences in the polished text.".format(sent_cnt, int_c)
            else:
                feedback = "The polished text contains {} sentences. However, it should contain {} sentences, please consider reducing the number of sentences in the polished text.".format(sent_cnt, int_c)
        return label, feedback

    def sentence_length_check(self, length, relation, polished_paragraph):
        label = 1
        pol_sent_list = self.split_into_sentences(polished_paragraph)
        pol_len_list = [len(i.split()) for i in pol_sent_list]
        int_l = int(length)
        feedback = "All sentences in the polished text contains {} {} words, it follows the user's requirement.".format(relation, int_l)
        if relation == "less than":
            for idx, tmp_len in enumerate(pol_len_list):
                if tmp_len >= int_l:
                    label = 0
                    feedback = "The sentence \"{}\" contains more than {} words, please consider reducing the number of words in this sentence.".format(pol_sent_list[idx], int_l)
                    break
        elif relation == "more than":
            for idx, tmp_len in enumerate(pol_len_list):
                if tmp_len <= int_l:
                    label = 0
                    feedback = "The sentence \"{}\" contains less than {} words, please consider increasing the number of words in this sentence.".format(pol_sent_list[idx], int_l)
                    break
        else:
            for idx, tmp_len in enumerate(pol_len_list):
                if tmp_len != int_l:
                    label = 0
                    if tmp_len < int_l:
                        feedback = "The sentence \"{}\" contains less than {} words, please consider increasing the number of words in this sentence.".format(pol_sent_list[idx], int_l)
                    else:
                        feedback = "The sentence \"{}\" contains more than {} words, please consider reducing the number of words in this sentence.".format(pol_sent_list[idx], int_l)
                    break
        return label, feedback
    ### tools end 

    def split_text_by_paragraphs(self, text):
        paragraphs = text.split('\n\n')
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        return paragraphs

    def calculate_by_paragraph(self, para_ori, para):
        all_g, all_f = 0, 0
        for para_idx in range(min(len(para_ori), len(para))):
            all_sent = re.split(r'(?<=[.!?])\s+', para[para_idx])
            all_sent_ori = re.split(r'(?<=[.!?])\s+', para_ori[para_idx])
            if len(all_sent) == len(all_sent_ori):
                cum_g = 0
                cum_f = 0
                for idx in range(len(all_sent)):
                    g, f, _ = self.some_model.score([all_sent_ori[idx]], [all_sent[idx]])
                    cum_g += g[0]
                    # print(g)
                    cum_f += f[0]
                tmp_g = cum_g / len(all_sent)
                tmp_f = cum_f / len(all_sent)
            else:
                g, f, _ = self.some_model.score([all_sent_ori], [all_sent])
                tmp_g = g[0]
                tmp_f = f[0]
            all_g += tmp_g
            all_f += tmp_f
        return all_g / (para_idx + 1), all_f / (para_idx + 1)

    def split_text_by_paragraphs_bartscore(self, text):
        paragraphs = text.split('\n\n')
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        return paragraphs
    
    def calculate_by_paragraph_bartscore(self, para_ori, para):
        all_score = 0
        for para_idx in range(min(len(para_ori), len(para))):
            all_sent = re.split(r'(?<=[.!?])\s+', para[para_idx])
            all_sent_ori = re.split(r'(?<=[.!?])\s+', para_ori[para_idx])
            if len(all_sent) == len(all_sent_ori):
                cum_score = 0
                for idx in range(len(all_sent)):
                    score = self.bart_scorer.score([all_sent_ori[idx]], [all_sent[idx]], batch_size=4)
                    cum_score += score[0]
                tmp_score = cum_score / len(all_sent)
            else:
                score = self.bart_scorer.score([" ".join(all_sent_ori)], [" ".join(all_sent)], batch_size=4)
                tmp_score = score[0]
            all_score += tmp_score
        return all_score / (para_idx + 1)

    def calculate_sentence_perplexity(self, sentence, model, tokenizer, device):
        inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_per_token = loss_per_token.view(shift_labels.size())
        return loss_per_token.sum(), len(loss_per_token[0])

    def calculate_paragraph_perplexity(self, paragraph, model_name='gpt2-large'):
        sentences = paragraph.split('. ')
        perplexities = []
        loss_list = []
        length_list = []
        for sentence in sentences:
            if sentence.strip():
                loss_sum, length = self.calculate_sentence_perplexity(sentence, 
                                                                      self.gpt2_model, 
                                                                      self.gpt2_tokenizer, 
                                                                      self.gpt2_device)
                loss_list.append(loss_sum)
                length_list.append(length)
        try:
            perplexities = torch.exp(sum(loss_list)/sum(length_list))
        except:
            perplexities = torch.tensor(-100)
        return perplexities

    def compute_cons_acc(self, ori_text, polished_paragraph, tool_lists):
        if len(tool_lists) == 1 and 'text_eval' in tool_lists[0]:
            accuracy = 1
            return "Constraints following feedback:\nAll the constraints have been satisfied.", accuracy
        cor_cnt, tot_cnt = 0, 0

        tmp_prompt_list = []
        for tool in tool_lists:
            if 'python' in tool:
                continue
            if 'text_eval' in tool:
                continue
            if 'keyword_keep_removal_check' in tool or 'sentence_modification_check' in tool:
                tool_call = 'self.' + tool.replace(")", ", polished_paragraph, ori_text)")
            else:
                tool_call = 'self.' + tool.replace(")", ", polished_paragraph)")
            if 'keyword_detection' in tool:
                continue
            try:
                tmp_output = eval(tool_call)
                tot_cnt += 1
                if 'word_count_check' in tool_call or 'keyword_frequency_check' in tool_call:
                    ind_output, freq, feedback = tmp_output
                    if ind_output:
                        cor_cnt += 1
                    tmp_prompt_list.append(feedback)
                else:
                    ind_output, feedback = tmp_output
                    if ind_output == 1:
                        cor_cnt += 1
                    tmp_prompt_list.append(feedback)
            except:
                tot_cnt += 1
                continue
        if tot_cnt == 0:
            accuracy = 1
        else:
            accuracy = cor_cnt / tot_cnt
        return "Constraints following feedback:\n" + "\n".join(tmp_prompt_list), accuracy
    
    def get_ppl_some_bart(self, init_content, polished_paragraph):
        ppl = self.calculate_paragraph_perplexity(polished_paragraph).cpu().item()
        some_g_f = self.calculate_by_paragraph(self.split_text_by_paragraphs(init_content), 
                                               self.split_text_by_paragraphs(polished_paragraph))
        some = some_g_f[0]
        bart = self.calculate_by_paragraph_bartscore(self.split_text_by_paragraphs_bartscore(init_content), 
                                                     self.split_text_by_paragraphs_bartscore(polished_paragraph))
        return ppl, some, bart

    def get_init_value(self, ori_text):
        ppl = self.calculate_paragraph_perplexity(ori_text).cpu().item()
        bart = self.calculate_by_paragraph_bartscore(self.split_text_by_paragraphs_bartscore(ori_text), 
                                                     self.split_text_by_paragraphs_bartscore(ori_text))
        # print("Initial PPL: {}".format(ppl))
        # print("Initial BART: {}".format(bart))
        return ppl, bart
    
    def get_ppl_bart(self, text):
        ppl, bart = self.get_init_value(text)
        return ppl, bart

    def get_rollout_reward(self, node):
        ###todo : need to change this
        ori_text, polished_paragraph, checking_tool_lists, ini_ppl, ini_bart = node.get_trajectory()

        # ini_ppl, ini_bart = self.get_init_value(ori_text=ori_text)
        # print(checking_tool_lists)
        # print("Inside get rollout reward")
        # print("Initial PPL: {}".format(ini_ppl))
        # print("Initial BART: {}".format(ini_bart))
        if polished_paragraph == "":
            ppl, some, bart = 0, 0, -100
            cons_acc = 0
        else:
            ppl, some, bart = self.get_ppl_some_bart(init_content=ori_text, 
                                                     polished_paragraph=polished_paragraph)
            cons_feedback, cons_acc = self.compute_cons_acc(ori_text=ori_text, 
                                                            polished_paragraph=polished_paragraph, 
                                                            tool_lists=checking_tool_lists)
            
        return ppl, some, bart, cons_acc, ini_ppl, ini_bart, cons_feedback