import threading
import openai
import time 
import json
import copy 
import re

from tetra_exp.ICL_selection.bart_score import BARTScorer
import sys
import json
import re
import nltk
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
import spacy
import nltk
from gliner import GLiNER

ner_model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")

def split_text_by_paragraphs_bartscore(text):
    paragraphs = text.split('\n\n')
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    return paragraphs

def calculate_by_paragraph_bartscore(para_ori, para):
    all_score = 0
    for para_idx in range(min(len(para_ori), len(para))):
        all_sent = re.split(r'(?<=[.!?])\s+', para[para_idx])
        all_sent_ori = re.split(r'(?<=[.!?])\s+', para_ori[para_idx])
        if len(all_sent) == all(all_sent_ori):
            cum_score = 0
            for idx in range(len(all_sent)):
                score = bart_scorer.score(all_sent_ori[idx], all_sent[idx], batch_size=4)
                cum_score += score[0]
            tmp_score = cum_score / len(all_sent)
        else:
            score = bart_scorer.score(" ".join(all_sent_ori), " ".join(all_sent), batch_size=4)
            tmp_score = score[0]
        all_score += tmp_score
    return all_score / (para_idx + 1)


import functools
import math
import os
import sys

from multiprocessing import Pool
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import (
    InputExample,
    InputFeatures,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    BertPreTrainedModel,
)

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
            device_str = 'cuda:{}'.format(0)

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

model_args = {'model_type': 'bert-base-cased',
            'g_dir': os.path.join(''),
            'f_dir': os.path.join(''),
            'm_dir': os.path.join(''),
            'weight_g': 0.55,
            'weight_f': 0.43,
            'weight_m': 0.02}
some_model = SOME(model_args)

import re

def split_text_by_paragraphs(text):
    paragraphs = text.split('\n\n')
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    return paragraphs
    # sentences = re.split(r'(?<=[.!?]) +', text)
    # final_sentences = []
    # for sent in sentences:
    #     if sent == '':
    #         continue
    #     final_sentences.append(sent)
    # return final_sentences

def calculate_by_paragraph(para_ori, para):
    all_g, all_f = 0, 0
    for para_idx in range(min(len(para_ori), len(para))):
        # all_sent = re.findall(r'\b\w+\b', para[para_idx])
        # all_sent_ori = re.findall(r'\b\w+\b', para_ori[para_idx])
        all_sent = re.split(r'(?<=[.!?])\s+', para[para_idx])
        all_sent_ori = re.split(r'(?<=[.!?])\s+', para_ori[para_idx])
        if len(all_sent) == all(all_sent_ori):
            cum_g = 0
            cum_f = 0
            for idx in range(len(all_sent)):
                g, f, _ = some_model.score(all_sent_ori[idx], all_sent[idx])
                cum_g += g[0]
                cum_f += f[0]
            tmp_g = cum_g / len(all_sent)
            tmp_f = cum_f / len(all_sent)
        else:
            g, f, _ = some_model.score(all_sent_ori, all_sent)
            tmp_g = g[0]
            tmp_f = f[0]
        all_g += tmp_g
        all_f += tmp_f
    return all_g / (para_idx + 1), all_f / (para_idx + 1)


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_sentence_perplexity(sentence, model, tokenizer, device):
    # Encode the sentence
    inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    
    # Calculate loss and perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # loss = outputs.loss
        logits = outputs.logits

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()  # Ignore the last token's prediction
        shift_labels = input_ids[..., 1:].contiguous()  # Ignore the first token for labels

        # Reshape logits and labels to be flat for loss calculation
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Reshape back to [batch_size, sequence_length]
        loss_per_token = loss_per_token.view(shift_labels.size())
    return loss_per_token.sum(), len(loss_per_token[0])

def calculate_paragraph_perplexity(paragraph, model_name='gpt2-large'):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Split paragraph into sentences (you can adjust the splitting logic)
    sentences = paragraph.split('. ')
    
    # Calculate perplexity for each sentence
    perplexities = []
    loss_list = []
    length_list = []
    for sentence in sentences:
        if sentence.strip():  # Skip empty sentences
            loss_sum, length = calculate_sentence_perplexity(sentence, model, tokenizer, device)
            loss_list.append(loss_sum)
            length_list.append(length)
    
    # Calculate the average perplexity
    try:
        perplexities = torch.exp(sum(loss_list)/sum(length_list))
    except:
        perplexities = torch.tensor(-100)
    # average_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('inf')
    return perplexities

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('')
model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained('')

def calculate_perplexity_chunks(text, max_length=512):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_tokens = 0

    # Tokenize the entire input text
    inputs = tokenizer(text, return_tensors='pt')['input_ids'][0]
    
    # Break input into chunks of max_length
    for i in range(0, len(inputs), max_length):
        chunk = inputs[i:i+max_length]
        if len(chunk) < 2:  # Skip very small chunks
            continue
        
        chunk = chunk.unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
            loss = outputs.loss
            total_loss += loss.item() * chunk.size(1)  # Multiply loss by number of tokens in chunk
            total_tokens += chunk.size(1)

    # Average the total loss over all tokens and compute perplexity
    average_loss = total_loss / total_tokens
    perplexity = np.exp(average_loss)
    return perplexity


import argparse
import copy
import glob
import json
import logging
import os
import re
import openai

# openai.api_base = "http://0.0.0.0:8000/v1"  # use the IP or hostname of your instance
# openai.api_key = "none"  # vLLM server is not authenticated

from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer, LlamaForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_txt_file(file_path):
    """
    Read a text file to string
    """
    with open(file_path, 'r') as file:
        return file.read()


def read_json(file_path):
    """
    Read a json file to dict
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def preprocess_text(text):
    """
    Clean up text: remove reference section, URLS, non-ascii chars
    """
    # clean up empty line
    paragraphs = text.split("\n")
    paragraphs = [i for i in paragraphs if len(i) > 0]
    # clean up section title and remove reference section
    cleaned_pargraphs = []
    for i in paragraphs:
        if i == "# References":
            break
        if i.startswith("#"):
            i = "section: " + i.replace("#", "").strip()
        cleaned_pargraphs.append(i)
    text = "\n".join(cleaned_pargraphs)
    # remove URLS
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove non-ascii char
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # remove citation bracket (e.g. [10])
    text = re.sub(r'\[\d+\]', '', text)
    # remove non alphanumeric char
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_conversation_prompt(filled_prompt):
    """
    From filled prompt, convert it into llama-2 conversation prompt
    """
    conv = get_conv_template("llama-2")
    conv.set_system_message("You are a fair evaluator language model.")
    conv.append_message(conv.roles[0], filled_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def format_prompt(prompt_template, rubric, response):
    """
    Fill prompt_template with rubric and response
    """
    prompt_template_copy = copy.deepcopy(prompt_template)
    data = copy.deepcopy(rubric)
    data.update({
        "instruction": f"You are a text editor. Your task is to improve the quality of this text.",
        "response": response
    })
    filled_prompt = prompt_template_copy.format(**data)
    return get_conversation_prompt(filled_prompt)

def get_grading_dict(responses,
                     model,
                     tokenizer,
                     prompt_template_path="",
                     rubric_path="",
                     temperature=0.01,
                     top_p=0.95,
                     max_new_tokens=512,
                     repetition_penalty=1.03,
                     logger=None):
    disable_sample = False
    grading = {}
    prompt_template = read_txt_file(prompt_template_path)
    rubrics = read_json(rubric_path)

    for response_idx, response in enumerate(responses):   
        grading[response_idx] = {}    
        for rubric_idx, rubric in enumerate(rubrics):
            # if logger is not None:
            #     logger.info(
            #         f"processing for rubric {rubric_idx + 1}/{len(rubrics)}, response {response_idx + 1}/{len(responses)}, response length: {len(response)}")
            print(f"processing for rubric {rubric_idx + 1}/{len(rubrics)}, response {response_idx + 1}/{len(responses)}, response length: {len(response)}")
            prompt = format_prompt(prompt_template=prompt_template, rubric=rubric, response=response)
            #### Replace the content here with vllm inference
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
            # geenrate output
            outputs = model.generate(input_ids,
                                     pad_token_id=tokenizer.eos_token_id,
                                     do_sample=not disable_sample,
                                     temperature=temperature,
                                     top_p=top_p,
                                     max_new_tokens=max_new_tokens,
                                     repetition_penalty=repetition_penalty)
            decoded_output = tokenizer.decode(outputs[0])

            decoded_output = decoded_output[decoded_output.find("[/INST]") + len("[/INST]"):].strip()
            feedback = decoded_output[:decoded_output.find("[RESULT]")]
            score = decoded_output[decoded_output.find("[RESULT]") + len("[RESULT]"):].replace("</s>", "").strip()
            try:
                int(score)
            except Exception as e:
                pattern = r"the overall score is (\d+)"
                match = re.search(pattern, feedback)
                if match:
                    score = match.group(1)

            grading[response_idx][rubric["criteria_description"]] = {"feedback": feedback, "score": score}
            # print("response_idx: {} finished.".format(response_idx))
    return grading

def replace_last_period(text):
    last_period_index = text.rfind('.')
    if last_period_index != -1:
        text = text[:last_period_index] + ':\n\n' + text[last_period_index + 1:]
    return text


# messages=[{'role': 'system', 'content': system_prompt},
#           {"role": "user", "content": user_input},
#           {'role': 'assistant', 'content': thought_action_1},
#           {'role': 'user', 'content': observation_1},
#           {'role': 'assistant', 'content': thought_action_2},
#           {'role': 'user', 'content': observation_2}]

def get_throught_action(tmp_message):
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=tmp_message,
        temperature=1.0,
        presence_penalty=0,
        frequency_penalty=0,
        )
    return response.choices[0].message.content

def get_action_and_tool(response):
    # Define patterns for each section
    thought_pattern = r"###THOUGHT:\n(.*?)\n###"
    tools_pattern = r"###TOOLS:\n(.*?)\n###"
    plan_pattern = r"###PLAN:\n(.*)"
    tools_list_pattern = r"\d+\.\s`([^`]+)`"
    tools_list = []
    thought_match = re.search(thought_pattern, response, re.DOTALL)
    tools_match = re.search(tools_pattern, response, re.DOTALL)
    plan_match = re.search(plan_pattern, response, re.DOTALL)
    # print(tools_match.group(1).strip())
    if thought_match:
        thought = thought_match.group(1).strip()
    if tools_match:
        tools = tools_match.group(1).strip().replace('```', '').replace("```", "")
        tools_list = [tool.strip() for tool in tools.splitlines() if tool.strip()]
    if plan_match:
        plan = plan_match.group(1).strip()
    
    return thought, tools_list, plan

def format_list(items):
    if not items:
        return ''
    elif len(items) == 1:
        return f'"{items[0]}"'
    else:
        quoted_items = [f'"{item}"' for item in items]
        initial_items = quoted_items[:-1]
        last_item = quoted_items[-1]
        formatted_string = ', '.join(initial_items)
        formatted_string += f', and {last_item}'
        return formatted_string
    
def remove_last_period(input_string):
    last_period_index = input_string.rfind('.')
    if last_period_index == -1:
        return input_string
    else:
        return input_string[:last_period_index] + input_string[last_period_index+1:]

def add_keyword_constraint(text):
    pronouns = {"i", "me", "my", "mine", "myself",
            "we", "us", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves"}
    pattern = r'Sentence: "(.*?)"; Improvement Plan: "(.*?)"'
    matches = re.findall(pattern, text, re.DOTALL)
    modified_entries = []
    for sentence, plan in matches:
        keywords = []
        token_lists = nltk.word_tokenize(sentence)
        labels = ["Person", "Award", "Date", "Competitions", "Company"]
        entities = ner_model.predict_entities(sentence, labels, threshold=0.5)
        for entity in entities:
            if entity["text"] not in keywords:
                if entity["text"].lower() not in pronouns:
                    keywords.append(entity["text"])
        ### add the LLM names
        llm_names = ["o1", "GPT-4o", "GPT-4", "GPT-3.5", "GPT-3", "Claude", "PaLM", "LLaMA", "Mistral", "Davinci", "Curie", "Babbage", "Ada"]
        for name in llm_names:
            if name in token_lists:
                if name not in keywords:
                    keywords.append(name)
        if keywords:
            constraint = format_list(keywords)
            # print(constraint)
            tmp_plan = remove_last_period(plan)
            new_plan = tmp_plan + ", keep the keywords " + constraint + "."
        else:
            new_plan = plan
        modified_entry = f'Sentence: "{sentence}"; Improvement Plan: "{new_plan}"'
        modified_entries.append(modified_entry)
    modified_text = '\n'.join(modified_entries)
    return modified_text

def get_gpt4_polish_response(user_input):
    polish_prompt = "You are a helpful writing assistant tasked with refining text."
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {'role': 'system', 'content': polish_prompt},
            {"role": "user", "content": user_input},
            ],
        temperature=1.0,
        presence_penalty=0,
        frequency_penalty=0,
        )
    return response.choices[0].message.content

def convert_to_paragraph(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip() != '']
    paragraph = ' '.join(lines)
    return paragraph

def get_gpt4_gram_feedback_response(user_input):
    grammar_check_prompt = "You are a helpful assistant tht helps to identify potential grammatical errors in a given text."
    grammar_pre_prompt = "Please analyze the following text for grammatical errors, including issues with sentence structure, punctuation, subject-verb agreement, tense consistency, pronoun usage, and any other common grammar mistakes. For each of the sentence that contain grammar errors, please format the output strictly as follows: 'Original: [original text]; Suggestion: [corrected text]'. Do not include any additional content. Text: "
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {'role': 'system', 'content': grammar_check_prompt},
            {"role": "user", "content": grammar_pre_prompt + user_input},
            ],
        temperature=1.0,
        presence_penalty=0,
        frequency_penalty=0,
        )
    return response.choices[0].message.content

def get_gpt4_fluency_feedback_response(user_input):
    fluency_check_prompt = "You are a helpful assistant tht helps to identify potential fluency problems in a given text."
    fluency_pre_prompt = "Please analyze the following text for fluency issues, including awkward phrasing, unnatural word choices, sentence flow, and readability problems. For each of the sentence that contain fluency problems, please format the output strictly as follows: 'Original: [original text]; Suggestion: [corrected text]'. If a sentence has no issues, do not include it in the output. Do not include any additional content. Text: "
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {'role': 'system', 'content': fluency_check_prompt},
            {"role": "user", "content": fluency_pre_prompt + user_input},
            ],
        temperature=1.0,
        presence_penalty=0,
        frequency_penalty=0,
        )
    return response.choices[0].message.content

def get_gpt4_coherence_feedback_response(user_input):
    coherence_check_prompt = "You are a helpful assistant tht helps to identify potential fluency problems in a given text."
    coherence_pre_prompt = "Please analyze the following text for coherence problems, such as unclear connections between ideas, lack of logical flow, abrupt transitions, or inconsistencies in the overall message. For each sentence or section that contains a coherence problem, format the output strictly as follows: 'Original: [original text]; Suggestion: [corrected text]'. If a sentence or section has no issues, do not include it in the output. Do not include any additional content. Text: "
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {'role': 'system', 'content': coherence_check_prompt},
            {"role": "user", "content": coherence_pre_prompt + user_input},
            ],
        temperature=1.0,
        presence_penalty=0,
        frequency_penalty=0,
        )
    return response.choices[0].message.content

def get_ppl_some_bart(init_content, polished_paragraph):
    ppl = calculate_paragraph_perplexity(polished_paragraph).cpu().item()
    some = calculate_by_paragraph(split_text_by_paragraphs(init_content), 
                                  split_text_by_paragraphs(polished_paragraph))
    bart = calculate_by_paragraph_bartscore(split_text_by_paragraphs_bartscore(init_content), 
                                            split_text_by_paragraphs_bartscore(polished_paragraph))
    return ppl, some, bart

def format_feedback(input_text):
    # if re.fullmatch(r"Original: .*?; Suggestion: .*?", input_text.strip()):
    #     return [input_text.strip()]
    pattern = r"Original: (.*?)(?:;|\n)Suggestion: (.*?)($|\nOriginal:)"
    matches = re.findall(pattern, input_text, re.DOTALL)
    standardized_output = []
    for original, suggestion, _ in matches:
        standardized_output.append(f"Original: {original.strip()}; Suggestion: {suggestion.strip()}")
    other_patten = r"Original: (.*?); Suggestion: (.*?)(?=(Original:|$))"
    other_matches = re.findall(other_patten, input_text, re.DOTALL)
    for original, suggestion, _ in other_matches:
        standardized_output.append(f"Original: {original.strip()}; Suggestion: {suggestion.strip()}")

    return standardized_output

def f1_score(list1, list2):
    # Convert lists to sets for easier calculation of intersection
    set1 = set(list1)
    set2 = set(list2)
    
    # Calculate true positives, precision, and recall
    true_positives = len(set1.intersection(set2))
    precision = true_positives / len(set1) if len(set1) > 0 else 0
    recall = true_positives / len(set2) if len(set2) > 0 else 0
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

from difflib import SequenceMatcher
def get_closest_match(target, options):
    closest_match = None
    highest_ratio = 0
    for option in options:
        similarity = SequenceMatcher(None, target, option).ratio()
        if similarity > highest_ratio:
            highest_ratio = similarity
            closest_match = option
    return closest_match

def tool_fb_prompt(tool_correct, tools_list, predefined_tool):
    if not tool_correct:
        tools_list_prefiex = [i.split("(")[0] for i in tools_list]
        correct_tools = set(predefined_tool)
        correct_tools_prefix = [i.split("(")[0] for i in correct_tools]

        feedback_parts = []
        for tool in tools_list:
            if tool in correct_tools:
                feedback_parts.append(f"Your tool call: `{tool}` is correct.")
            else:
                closest_match = get_closest_match(tool.split("(")[0], correct_tools)
                if closest_match:
                    if closest_match.split("(")[0] == tool.split("(")[0]:
                        feedback_parts.append(f"Your tool call: `{tool}` is wrong, it should be `{closest_match}`.")
                    else:
                        feedback_parts.append(f"Your tool call: `{tool}` is wrong and unnecessary, it should be dropped.")
                else:
                    feedback_parts.append(f"Your tool call: `{tool}` is wrong and unnecessary, it should be dropped.")
        for cor_tool in correct_tools:
            if cor_tool.split("(")[0] not in tools_list_prefiex:
                feedback_parts.append(f"You have missed this tool call: `{cor_tool}`, it should be included.")
        feedback = "Tool usage feedback:\n" + "\n".join(feedback_parts)
    else:
        tools_string = "; ".join(f"`{tool}`" for tool in tools_list)
        feedback = f"Tool usage feedback:\nThe tools are used correctly please keep using it: {tools_string}"
    return feedback

def create_feedback_prompt(flu_fb, coh_fb, gra_fb):
    all_gra_fb = "\n".join(gra_fb)
    all_flu_fb = "\n".join(flu_fb)
    all_coh_fb = "\n".join(coh_fb)
    return "Text quality feedback:\nGrammatical Feedback: {}\n\nCoherence Feedback: {}\n\nFluency Feedback: Original: {}".format(all_gra_fb, all_coh_fb, all_flu_fb)

#### need to define the list of quality checking functions here
def word_count_check(count, relation, polished_paragraph):
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

def keyword_frequency_check(keyword, frequency, relation, polished_paragraph):
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

def keyword_keep_removal_check(keyword, relation, polished_paragraph, ori_text):
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
            feedback = "The keyword \"{}\" has been removed some of the sentences, please conside adding them back.".format(keyword)
    else: ### "remove"
        if keyword not in polished_paragraph:
            label = 1
            feedback = "The keyword \"{}\" has been removed, it follows the user's requirement.".format(keyword)
        else:
            feedback = "The keyword \"{}\" exist in the text, please conside removing it.".format(keyword)
    return label, feedback

def split_into_sentences(paragraph):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    sentences = [re.sub(r'[.!?]$', '', sentence) for sentence in sentences]
    return sentences

def sentence_modification_check(sentence_id, relation, polished_paragraph, ori_text):
    ori_sent_list = split_into_sentences(ori_text)
    sent_list = []
    if relation == "change":
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
        for sent_id in sentence_id:
            ori_sent = ori_sent_list[sent_id - 1]
            if ori_sent not in polished_paragraph:
                feedback = "The sentence '{}' has been changed, please do not modify this sentence.".format(ori_sent)
                return 0, feedback
            sent_list.append(ori_sent)
        sentences = "; ".join(sent_list)
        feedback = "The targeted sentence has not been changed, it follows the user's requirement. Please keep them unchanged: {}".format(sentences)
        return 1, feedback

def sentence_count_check(count, relation, polished_paragraph):
    label = 0
    polished_sent_list = split_into_sentences(polished_paragraph)
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

def sentence_length_check(length, relation, polished_paragraph):
    label = 1
    pol_sent_list = split_into_sentences(polished_paragraph)
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

def cons_fb_prompt(ori_text, polished_paragraph, predefined_tool):
    if len(predefined_tool) == 1 and 'text_eval' in predefined_tool[0]:
        accuracy = 1
        return "Constraints following feedback:\nAll the constraints have been satisfied.", accuracy
    cor_cnt, tot_cnt = 0, 0

    tmp_prompt_list = []
    for tool in predefined_tool:
        if 'text_eval' in tool:
            continue
        if 'keyword_keep_removal_check' in tool or 'sentence_modification_check' in tool:
            tool_call = tool.replace(")", ", polished_paragraph, ori_text)")
        else:
            tool_call = tool.replace(")", ", polished_paragraph)")
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
    accuracy = cor_cnt / tot_cnt
    return "Constraints following feedback:\n" + "\n".join(tmp_prompt_list), accuracy

# def cons_fb_prompt(ori_text, polished_paragraph, predefined_tool):
#     if len(predefined_tool) == 1 and 'text_eval' in predefined_tool[0]:
#         accuracy = 1
#         return "All the constraints have been satisfied.", accuracy
#     cor_cnt, tot_cnt = 0, 0

#     tmp_prompt_list = []
#     for tool in predefined_tool:
#         if 'text_eval' in tool:
#             continue
#         # tool_call = tool.replace(")", ", polished_paragraph)")
#         if 'keyword_keep_removal_check' in tool or 'sentence_modification_check' in tool:
#             tool_call = tool.replace(")", ", polished_paragraph, ori_text)")
#             # tool_call = tool.replace(")", ", {}, {})".format(polished_paragraph, ori_text))
#         else:
#             tool_call = tool.replace(")", ", polished_paragraph)")
#             # tool_call = tool.replace(")", ", {})".format(polished_paragraph))
#         tmp_output = eval(tool_call)
#         tot_cnt += 1
#         if 'word_count_check' in tool_call or 'keyword_frequency_check' in tool_call:
#             # tmp_output = (376, 1)
#             output, freq = tmp_output
#             if output:
#                 tmp_prompt_list.append("The constraint {} has been satisfied, please keep following it.".format(tool))
#                 cor_cnt += 1
#             else:
#                 if 'keyword_frequency_check' in tool_call:
#                     cons_freq = tool_call.split(", ")[1]
#                     tmp_prompt_list.append("The constraint {} has not been satisfied. The required frequency is {}, your output frequency is {}".format(tool, cons_freq, freq))
#                 else:
#                     cons_cnt = tool_call.split(", ")[0].replace("word_count_check(", "")
#                     tmp_prompt_list.append("The constraint {} has not been satisfied. The required word count is {}, your output word count is {}".format(tool, cons_cnt, freq))
#         else:
#             if tmp_output == 1: ## is correct
#                 tmp_prompt_list.append("The constraint {} has been satisfied, please keep following it.".format(tool))
#                 cor_cnt += 1
#             else:
#                 tmp_prompt_list.append("The constraint {} has not been satisfied, please re-consider your plan to follow the requirement.".format(tool))
#     accuracy = cor_cnt / tot_cnt
#     return " ".join(tmp_prompt_list), accuracy

def ob_prompt(polished_para, tool_acc_fb, text_quality_fb, text_cons_fb):
    prompt = "###OBSERVATION: The polished text:\n{}\n\nHere are the feedback:\n{}\n\n{}\n\n{}\n\nPlease use these feedback to further refine the text by generating new text refinement plans and tool usages according to the response format.".format(polished_para, tool_acc_fb, text_quality_fb, text_cons_fb)
    return prompt

def generate_final_prompt(input):
    editing_plan = input.split('###PLAN:')[-1].replace("\n\n", "\n").replace("*", "").lstrip('\n')
    tmp_cleaned_text = re.sub(r'^\s+(Sentence:)', r'\1', editing_plan, flags=re.MULTILINE)
    cleaned_text = add_keyword_constraint(tmp_cleaned_text)
    ending_prompt = """### INSTRUCTIONS:

Using the information provided in each text editing plan (### INPUT), generate the polished version of each sentence by applying the specified improvements. Maintain the original order of sentences.

**In your output, provide only the final polished sentences, one after another, without any prefixes, numbering, or additional text.**

### INPUT:\n\n"""

    final_prompt = ending_prompt + cleaned_text + "\n\n### OUTPUT:\n\n"
    return final_prompt


def check_cons(cons_list):
    if len(set(cons_list)) == 1:
        return None
    max_element = max(cons_list)
    indices = [i+1 for i, x in enumerate(cons_list) if x == max_element]
    return indices

def early_stop(ppl_list, some_list, bart_list, tool_list, cons_list, patience=2):
    cons_step = check_cons(cons_list)
    if cons_step is not None:
        if len(cons_step) == 1:
            return cons_step[0]
        elif len(cons_step) > 1:
            selected_step = 0
            best_score = -100
            for tmp_step in cons_step:
                tmp_ppl = ppl_list[tmp_step]
                tmp_some = some_list[tmp_step]
                tmp_bart = bart_list[tmp_step]
                tot_score = (ppl_list[0] - tmp_ppl) + 100 * (tmp_some - some_list[0]) + (tmp_bart - bart_list[0])
                if tot_score - best_score > 1e-3:
                    selected_step = tmp_step
                    best_score = tot_score
                elif tot_score - best_score < 1e-3 and tot_score - best_score > 0:
                    tool_tmp_step = tmp_step - 1
                    tmp_tool = tool_list[tool_tmp_step]
                    prev_tmp_tool = tool_list[selected_step - 1]
                    if tmp_tool > prev_tmp_tool:
                        selected_step = tmp_step
            return selected_step
    else:
        wait_ppl, wait_some, wait_bart = 0, 0, 0
        best_ppl, best_some, best_bart = ppl_list[0], some_list[0], bart_list[0]
        best_ppl_idx, best_some_idx, best_bart_idx = 0, 0, 0
        for epoch in range(0, len(ppl_list)):
            current_ppl, current_some, current_bart = ppl_list[epoch], some_list[epoch], bart_list[epoch]
            if current_ppl >= best_ppl + 1e-2:
                wait_ppl += 1
            else:
                best_ppl = current_ppl
                best_ppl_idx = epoch
                wait_ppl = 0
            if current_some - best_some <= 1e-3:
                wait_some += 1
            else:
                best_some = current_some
                best_some_idx = epoch
                wait_some = 0 
            if current_bart - best_bart <= 1e-3:
                wait_bart += 1
            else:
                best_bart = current_bart
                best_bart_idx = epoch
                wait_bart = 0
            if wait_ppl >= patience and wait_bart >= patience and wait_bart >= patience:
                best_epoch = max(best_ppl_idx, best_bart_idx, best_some_idx)
                return best_epoch
        return None  # No early stopping triggered

##todo: need to add the function to call 'identify_keyword()' to automatically presenve the keywords
def call_davinci(item, thread_results, thread_id):
    # ppl_list = []
    # some_score_list = []
    # bart_score_list = []
    # try:
        constrain_instruction = item['final_instruction']
        instruct2gpt4o = replace_last_period(constrain_instruction)
        ppl_list = [calculate_paragraph_perplexity(item['content']).cpu().item()]
        some_score_list = [0]
        bart_score_list = [-100]
        tool_f1_list = []
        cons_acc_list = []

        init_content = convert_to_paragraph(item['content'])
        init_input = instruct2gpt4o + item['content']
        new_message = copy.deepcopy(message)
        ### get the first response
        new_message.append({'role': 'user', 'content': init_input})
        final_message = [message[0],
                        {'role': 'user', 'content': init_input}]
        keep_improve = True
        iter_cnt = 0
        step = None
        # tool_list = []
        ### begin the iterative refinement
        while keep_improve:
            thought_action = get_throught_action(new_message)
            ### extract the tools, thought and plan
            thought, tools_list, plan = get_action_and_tool(thought_action)
            polished_text = get_gpt4_polish_response(generate_final_prompt(plan))
            polished_paragraph = convert_to_paragraph(polished_text)
            ### evaluate the text quality according to ppl, some, bart
            p, s, b = get_ppl_some_bart(init_content, polished_paragraph)
            ppl_list.append(p)
            some_score_list.append(s[0])
            bart_score_list.append(b)
            
            ### evaluate the tool accuracy
            if iter_cnt == 0:
                predefined_tool = []
                if 'constraint_info' in item.keys():
                    for i in item['constraint_info']:
                        tmp_list = i['function_call'].split(";")
                        for sub_i in tmp_list:
                            predefined_tool.append(sub_i.strip())
                    # predefined_tool = [i['function_call'] for i in item['constraint_info']]
                predefined_tool.append('text_eval()')
            if 'text_eval()' not in tools_list:
                tools_list.append('text_eval()')
            tool_f1 = f1_score(tools_list, predefined_tool)
            tool_f1_list.append(tool_f1)
            if abs(tool_f1 - 1) < 1e-3:
                tool_correct = True
            else:
                tool_correct = False

            fluency_feedback = format_feedback(get_gpt4_fluency_feedback_response(polished_paragraph))
            coherence_feedback = format_feedback(get_gpt4_coherence_feedback_response(polished_paragraph))
            grammar_feedback = format_feedback(get_gpt4_gram_feedback_response(polished_paragraph))
            text_quality_fb = create_feedback_prompt(fluency_feedback, coherence_feedback, grammar_feedback)
            
            tool_acc_fb = tool_fb_prompt(tool_correct, tools_list, predefined_tool)
            text_cons_fb, cons_acc = cons_fb_prompt(init_content, polished_paragraph, predefined_tool)
            cons_acc_list.append(cons_acc)
            # print(text_cons_fb)
            # print('==============')
            observation = ob_prompt(polished_paragraph, tool_acc_fb, text_quality_fb, text_cons_fb)
            ### add the thought_action and observation into the message
            new_message.append({'role': 'assistant', 'content': thought_action})
            new_message.append({'role': 'user', 'content': observation})

            final_message.append({'role': 'assistant', 'content': thought_action})
            final_message.append({'role': 'user', 'content': observation})
            iter_cnt += 1
            ### check if should keep improve or not
            ## 1) max_iter has reached (==4); 2) tool usage accuracy is not improved; 3) text fluency & coherence & gram has not been improved
            if iter_cnt == 5:
                keep_improve = False
                # generated_message = copy.deepcopy(final_message)
            step = early_stop(ppl_list=ppl_list, some_list=some_score_list, bart_list=bart_score_list, tool_list=tools_list, cons_list=cons_acc_list)
            if step:
                keep_improve = False
                # diff = -(2 * (iter_cnt - step))
                # generated_message = copy.deepcopy(final_message[:diff])
                # tmp_ob = copy.deepcopy(generated_message[-1])
                # # print(tmp_ob)
                # # print()
                # # print(tmp_ob['content'])
                # # print("==========================")
                # final_ob = tmp_ob['content'].split('Here are the feedback:\n')[0] + "Here are the feedback:\nThe generated response adheres to the user's instructions and constraints and maintains a high standard of quality."
                # generated_message[-1]['content'] = final_ob
            if keep_improve == False:
                break
        item['full_message'] = final_message
        # item['generated_message'] = generated_message
        item['score_list'] = {'ppl': ppl_list,
                            'some': some_score_list, 
                            'bart': bart_score_list,
                            'tool_f1': tool_f1_list,
                            'cons_acc': cons_acc_list,
                            }
        item['early_stop'] = step
        item['tot_iter_cnt'] = iter_cnt
        thread_results[thread_id] = item
    # except:
    #     thread_results[thread_id] = None


import random
import json

parallel_num = 10
MODEL_NAME=''
openai.base_url = ''
openai.api_key = ''

icl_content = open("").readlines()
message = []
for tmp_item in icl_content:
    message.append(json.loads(tmp_item))

task_input = {
    'no_constraint': [''],
    'one_constraint': [''],
    'two_constraint': [''],
    'three_constraint': [''],
    }

task_output_name = {
    'no_constraint': [''],
    'one_constraint': [''],
    'two_constraint': [''],
    'three_constraint': [''],
    }

idx = 0
left_data = []

# for times in range(5):
for task_name in task_input.keys():
    print(task_name)
    if task_name == 'no_constraint':
        icl_content = open("").readlines()
        message = []
        for tmp_item in icl_content:
            message.append(json.loads(tmp_item))
    else:
        icl_content = open("").readlines()
        message = []
        for tmp_item in icl_content:
            message.append(json.loads(tmp_item))

    # output_file_name = task_output_name[task_name][0].replace(".jsonl", ".{}.jsonl".format(times))
    output_file_name = task_output_name[task_name][0]
    input_data = open(task_input[task_name][0]).readlines()
    print(output_file_name)
    print(task_input[task_name][0])
    print("=============================")
    for idx, tmp_input in enumerate(input_data):
        tmp_dict = json.loads(tmp_input)
        # print(tmp_dict)
        # print(type(tmp_dict))
        idx_name = "{}_{}".format(idx, task_name)
        gpt4o_polish_prompt = ""
        if 'gpt4o_polish_prompt' in tmp_dict.keys():
            gpt4o_polish_prompt = tmp_dict['gpt4o_polish_prompt']
        gpt4o_polish_response = ""
        if 'gpt4o_polish_response' in tmp_dict.keys():
            gpt4o_polish_response = tmp_dict['gpt4o_polish_response']
        info = ""
        if 'info' in tmp_dict.keys():
            info = tmp_dict['info']
        final_instruction = ""
        if 'final_instruction' in tmp_dict.keys():
            final_instruction = tmp_dict['final_instruction']

        constraint_info = []
        if 'constraint_info' in tmp_dict.keys():
            for ind_info in tmp_dict['constraint_info']:
                constraint_info.append({'prompt': ind_info['prompt'],
                                        'constrain_type': ind_info['constrain_type'],
                                        'value': ind_info['value'],
                                        'function_call': ind_info['function_call'].replace("sentence_check", "sentence_modification_check")
                                        })

        tmp_left_dict = {
            'id': idx_name,
            'source': tmp_dict['source'],   ### where does this data come from
            'content': tmp_dict['content'], ### the source sentence of this data
            'gpt4o_polish_prompt': gpt4o_polish_prompt,
            'gpt4o_polish_response': gpt4o_polish_response,
            'info': info,
            'final_instruction': final_instruction,
            'constraint_info': constraint_info,
        }
        left_data.append(tmp_left_dict)
        # if idx == 0:
        #     break
    print("Length of the input data: {}".format(len(left_data)))

    ### multi-processing begins here:
    while True:
        error_items = []
        for i in range(0, len(left_data), parallel_num):
            c_data_for_gen = []
            for j in range(parallel_num):
                if i + j <= len(left_data)-1:
                    c_data_for_gen.append(left_data[i+j])

            thread_results = {}
            threads = []

            for thread_id, item in enumerate(c_data_for_gen):
                t = threading.Thread(target=call_davinci, args=(item, thread_results, thread_id))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
        
            c_data_to_save = []
            for thread_id in range(len(c_data_for_gen)):
                if thread_id in thread_results and thread_results[thread_id] != None:
                    c_data_to_save.append(thread_results[thread_id])
                else:
                    error_items.append(c_data_for_gen[thread_id])
            with open(output_file_name, "a") as outfile:
                for entry in c_data_to_save:
                    json.dump(entry, outfile)
                    outfile.write("\n")
            print(f'{len(c_data_to_save)} are generated ...')

            # time_gap = 5
            # for _ in range(time_gap):
                # time.sleep(1)
        if len(error_items) == 0:
            break
        else:
            left_data = error_items