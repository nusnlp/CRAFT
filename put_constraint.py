import os
import random 
import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.base_url = ''
openai.api_key = ''
'''
sentence constraint:
    a) don't change a certain sentence
    b) change a certain sentence
    c) don't change a certain number of sentence
    d) change a certain number of sentence
'''

def unchange_constraint(input_dict):
    unchange_list = input_dict['info']['unchange_info']['identical_sent_idx']
    if not unchange_list:
        sent_len_list = input_dict['info']['len_info']['src_sent_lens']
        max_len = max(sent_len_list)
        sent_idx = sent_len_list.index(max_len)
    else:
        sent_idx = random.choice(unchange_list)
    prompt = "Do not change the {}th sentence.".format(sent_idx + 1)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'sent_unchange_constraint',
        'value': {'cnt': [sent_idx + 1]},
        'function_call': 'sentence_check([{}], "unchange")'.format(sent_idx + 1),
    }
    return constraint

def change_constraint(input_dict):
    sent_cnt = input_dict['info']['len_info']['src_sent_cnt']
    all_list = [i for i in range(sent_cnt)]
    sent_idx = random.choice(all_list)
    prompt = "Only change {}th sentence.".format(sent_idx + 1)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'sent_change_constraint',
        'value': {'cnt': [sent_idx + 1]},
        'function_call': 'sentence_check([{}], "change")'.format(sent_idx + 1),
    }
    return constraint

def unchange_multi_constraint(input_dict):
    unchange_list = input_dict['info']['unchange_info']['identical_sent_idx']
    sent_len_list = [i for i in range(len(input_dict['info']['len_info']['src_sent_lens']))]
    # sent_len_list = input_dict['info']['len_info']['src_sent_lens']
    if len(unchange_list) <= 3 and len(sent_len_list) <= 3:
        return None
    elif len(sent_len_list) > 3:
        tmp_sent_idx = random.sample(sent_len_list, 3)
    else:    
        tmp_sent_idx = random.sample(unchange_list, 3)
    sent_idx = sorted(tmp_sent_idx)
    prompt = "Do not change the {}th, and {}th sentence.".format(sent_idx[0] + 1, sent_idx[1] + 1)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'unchange_constraint',
        'value': {'cnt': [sent_idx[0] + 1, sent_idx[1] + 1]},
        'function_call': 'sentence_check([{}, {}], "unchange")'.format(sent_idx[0] + 1, sent_idx[1] + 1),
    }
    return constraint

def change_multi_constraint(input_dict):
    sent_cnt = input_dict['info']['len_info']['src_sent_cnt']
    all_list = [i for i in range(sent_cnt)]
    if len(all_list) <= 3:
        return None
    sent_idx = random.sample(all_list, 3)
    prompt = "Only change {}th, {}th and {}th sentence.".format(sent_idx[0] + 1, sent_idx[1] + 1, sent_idx[2] + 1)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'change_constraint',
        'value': {'cnt': [sent_idx[0] + 1, sent_idx[1] + 1, sent_idx[2] + 1]},
        'function_call': 'sentence_check([{}, {}, {}], "change")'.format(sent_idx[0] + 1, sent_idx[1] + 1, sent_idx[2] + 1),
    }
    return constraint

'''
length constraint: 
    a) total length at least n
    b) total length no more than n
    c) number of sentence less than n
    d) number of sentence more than n
    e) sentence length at least n
    f) sentence length no more than n
'''

def round_to(value, level):
    tmp = value // 20
    if level == 'upper':
        return 20 * (tmp + 1)
    else:
        return max(10, 20 * tmp)

def tot_length_more_than_constraint(input_dict):
    sent_cnt_list = input_dict['info']['len_info']['tgt_sent_lens']
    tot_len = sum(sent_cnt_list)
    cnt = round_to(tot_len, 'lower')
    prompt = "Output contain more than {} tokens.".format(cnt)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'tot_len_more_than',
        'value': {'cnt': cnt},
        'function_call': 'word_count_check({}, "more than")'.format(cnt),
    }
    return constraint

def tot_length_less_than_constraint(input_dict):
    sent_cnt_list = input_dict['info']['len_info']['tgt_sent_lens']
    tot_len = sum(sent_cnt_list)
    cnt = round_to(max(10, tot_len + 20), "upper")
    prompt = "Output contain less than {} tokens.".format(cnt)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'tot_len_less_than',
        'value': {'cnt': cnt},
        'function_call': 'word_count_check({}, "less than")'.format(cnt),
    }
    return constraint

def tot_length_range_constraint(input_dict):
    sent_cnt_list = input_dict['info']['len_info']['tgt_sent_lens']
    tot_len = sum(sent_cnt_list)
    # cnt = max(10, tot_len - 10)
    new_cnt = round_to(max(10, tot_len - 10), "lower")
    max_cnt = new_cnt + 20
    prompt = "Output contain less than {} tokens and more than {} tokens.".format(max_cnt, new_cnt)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'tot_len_range',
        'value': {'max_cnt': max_cnt,
                  'min_cnt': new_cnt},
        'function_call': 'word_count_check({}, "less than"); word_count_check({}, "more than")'.format(max_cnt, new_cnt),
    }
    return constraint

def sentence_count_more_than_constraint(input_dict):
    sent_cnt_list = input_dict['info']['len_info']['tgt_sent_lens']
    tot_sent = len(sent_cnt_list)
    cnt = max(1, tot_sent - 2)
    prompt = "Output contain more than {} sentences.".format(cnt)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'sent_cnt_more_than',
        'value': {'cnt': cnt},
        'function_call': 'sentence_count_check({}, "more than")'.format(cnt),
    }
    return constraint

def sentence_count_less_than_constraint(input_dict):
    sent_cnt_list = input_dict['info']['len_info']['tgt_sent_lens']
    tot_sent = len(sent_cnt_list)
    cnt = tot_sent + 2
    prompt = "Output contain less than {} sentences.".format(cnt)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'sent_cnt_less_than',
        'value': {'cnt': cnt},
        'function_call': 'sentence_count_check({}, "less than")'.format(cnt),
    }
    return constraint

def sent_length_more_than_constraint(input_dict):
    sent_cnt_list = input_dict['info']['len_info']['tgt_sent_lens']
    # tot_sent = len(sent_cnt_list)
    # max_len = max(sent_cnt_list)
    min_len = min(sent_cnt_list)
    # tot_len = sum(sent_cnt_list)
    cnt = min_len
    prompt = "Each sentence contain more than {} tokens.".format(cnt)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'per_len_more_than',
        'value': {'cnt': cnt},
        'function_call': 'sentence_length_check({}, "more than")'.format(cnt),
    }
    return constraint

def sent_length_less_than_constraint(input_dict):
    sent_cnt_list = input_dict['info']['len_info']['tgt_sent_lens']
    # tot_sent = len(sent_cnt_list)
    max_len = max(sent_cnt_list)
    # min_len = min(sent_cnt_list)
    # tot_len = sum(sent_cnt_list)
    cnt = max(5, max_len - 10)
    prompt = "Each sentence contain less than {} tokens.".format(cnt)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'per_len_less_than',
        'value': {'cnt': cnt},
        'function_call': 'sentence_length_check({}, "less than")'.format(cnt),
    }
    return constraint

'''
keyword constraint: 
    a) keep *** words
    b) remove *** words
    c) occur *** times
    d) occur more than *** times
    e) occur less than *** times
'''
def keep_keyword_constraint(input_dict):
    ner_info = input_dict['info']['keyword_info']['src_ner_info']
    keyword_info = input_dict['info']['keyword_info']['src_keyword_info']
    keyword_list = []
    for ner in ner_info:
        keyword_list.append(ner['word'])
    for item in keyword_info:
        keyword_list.append(item[0])
    keyword = random.choice(keyword_list)
    prompt = "Do not change the word \'{}\'.".format(keyword)
    times = input_dict['source'].count(keyword)
    constraint = {
        'prompt': prompt,
        'constrain_type': 'keep_keyword',
        'value': {'keyword': keyword,
                  'cnt': times},
        'function_call': 'keyword_keep_removal_check("{}", "keep")'.format(keyword),
    }
    return constraint

def remove_keyword_constraint(input_dict):
    ner_info = input_dict['info']['keyword_info']['src_ner_info']
    keyword_info = input_dict['info']['keyword_info']['src_keyword_info']
    keyword_list = []
    for ner in ner_info:
        keyword_list.append(ner['word'])
    for item in keyword_info:
        keyword_list.append(item[0])
    try:
        keyword = random.choice(keyword_list)
        prompt = "Do not use the word \'{}\'.".format(keyword)
        constraint = {
            'prompt': prompt,
            'constrain_type': 'remove_keyword',
            'value': {'keyword': keyword,
                    'cnt': 0},
            'function_call': 'keyword_keep_removal_check("{}", "remove")'.format(keyword),
        }
        return constraint
    except:
        return None

def keyword_frequency_constraint(input_dict):
    ner_info = input_dict['info']['keyword_info']['tgt_ner_info']
    keyword_info = input_dict['info']['keyword_info']['tgt_keyword_info']
    keyword_list = {}
    for ner in ner_info:
        keyword_list[ner['word']] = 1
    for item in keyword_info:
        keyword_list[item[0]] = item[1]
    try:
        keyword = random.choice(keyword_list.keys())
        times = keyword_list[keyword]
        # times = random.choice([i for i in range(min_time, max_time)])
        prompt = "The word \'{}\' should appear {} times.".format(keyword, times)
        constraint = {
            'prompt': prompt,
            'constrain_type': 'keyword_freq',
            'value': {'keyword': keyword,
                    'cnt': times},
            'function_call': 'keyword_frequency_check("{}", {}, "equal")'.format(keyword, times),
        }
        return constraint
    except:
        return None

def keyword_occur_more_than_constraint(input_dict):
    ner_info = input_dict['info']['keyword_info']['tgt_ner_info']
    keyword_info = input_dict['info']['keyword_info']['tgt_keyword_info']
    keyword_list = {}
    for ner in ner_info:
        keyword_list[ner['word']] = 1
    for item in keyword_info:
        keyword_list[item[0]] = item[1]
    try:
        keyword = random.choice(keyword_list.keys())
        times = keyword_list[keyword] - 1
        prompt = "The word \'{}\' should appear at least {} times.".format(keyword, times)
        constraint = {
            'prompt': prompt,
            'constrain_type': 'keyword_freq_more',
            'value': {'keyword': keyword,
                    'cnt': times},
            'function_call': 'keyword_frequency_check("{}", {}, "more than")'.format(keyword, times),
        }
        return constraint
    except:
        return None

def keyword_occur_less_than_constraint(input_dict):
    ner_info = input_dict['info']['keyword_info']['tgt_ner_info']
    keyword_info = input_dict['info']['keyword_info']['tgt_keyword_info']
    keyword_list = {}
    for ner in ner_info:
        keyword_list[ner['word']] = 1
    for item in keyword_info:
        keyword_list[item[0]] = item[1]
    try:
        keyword = random.choice(keyword_list.keys())
        times = keyword_list[keyword] + 1
        prompt = "The word \'{}\' should appear less than {} times.".format(keyword, times)
        constraint = {
            'prompt': prompt,
            'constrain_type': 'keyword_freq_less',
            'value': {'keyword': keyword,
                    'cnt': times},
            'function_call': 'keyword_frequency_check("{}", {}, "less than")'.format(keyword, times),
        }
        return constraint
    except:
        return None

def simple_prompt(task_instruct, constraints):
    return task_instruct + " " + constraints


def polish_prompt(prompt_list):
    new_prompt = " ".join(prompt_list)
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant to polish sentence in natural language.'},
            {"role": "user", "content": 'Please rewrite the following text to be more fluent, without changing the original meaning. You should directly output the revised text. The original text:\n' + new_prompt},
            ],
        temperature=0.7,
        presence_penalty=0,
        frequency_penalty=0,
        )
    msg = response.choices[0].message.content
    return msg

sent_constraint_list = [unchange_constraint, unchange_multi_constraint]
length_constraint_list = [tot_length_more_than_constraint, tot_length_less_than_constraint, tot_length_range_constraint]
sent_length_constraint_list = [sentence_count_less_than_constraint, sentence_count_more_than_constraint]
sent_count_constraint_list = [sent_length_more_than_constraint, sent_length_less_than_constraint]
keyword_constraint_list = [keep_keyword_constraint, remove_keyword_constraint, 
                   keyword_frequency_constraint, keyword_occur_more_than_constraint, 
                   keyword_occur_less_than_constraint]

constraint_list = [sent_constraint_list, length_constraint_list, sent_length_constraint_list, sent_count_constraint_list, keyword_constraint_list]

import json
input_path = ""
input_data = open(input_path).readlines()

### four

cnt = 0
final_instruct_list = []
for item in input_data:
    item_dict = json.loads(item)
    source = item_dict['source']
    content = item_dict['content']
    gpt4o_polish_prompt = item_dict['gpt4o_polish_prompt']
    gpt4o_polish_response = item_dict['gpt4o_polish_response']
    info = item_dict['info']
    constraint_func = random.sample(constraint_list, 4)
    task_instruct = "Polish the following text."
    # constraint_ins = constraint_func(item_dict)
    constraint_ins = []
    for sub_area in constraint_func:
        tmp_func = random.choice(sub_area)
        tmp_constraint = tmp_func(input_dict=item_dict)
        constraint_ins.append(tmp_constraint)

    if constraint_ins:
        prompt_list = [task_instruct]
        for ind_constraint in constraint_ins:
            if ind_constraint:
                prompt_list.append(ind_constraint['prompt'])
        final_prompt = polish_prompt(prompt_list)

        final_instruct_list.append({
            'id': cnt,
            'source': source,
            'content': content, 
            'gpt4o_polish_prompt': gpt4o_polish_prompt,
            'gpt4o_polish_response': gpt4o_polish_response,
            'info': info,
            'final_instruction': final_prompt,
            'constraint_info': constraint_ins
        })
        cnt += 1
    if cnt % 50 == 0:
        print(cnt)
        
with open('', 'w', encoding='utf-8') as file:
    for entry in final_instruct_list:
        file.write(json.dumps(entry) + '\n')

### three

cnt = 0
final_instruct_list = []
for item in input_data:
    item_dict = json.loads(item)
    source = item_dict['source']
    content = item_dict['content']
    gpt4o_polish_prompt = item_dict['gpt4o_polish_prompt']
    gpt4o_polish_response = item_dict['gpt4o_polish_response']
    info = item_dict['info']
    constraint_func = random.sample(constraint_list, 3)
    task_instruct = "Polish the following text."
    # constraint_ins = constraint_func(item_dict)
    constraint_ins = []
    for sub_area in constraint_func:
        tmp_func = random.choice(sub_area)
        tmp_constraint = tmp_func(input_dict=item_dict)
        constraint_ins.append(tmp_constraint)

    if constraint_ins:
        prompt_list = [task_instruct]
        for ind_constraint in constraint_ins:
            if ind_constraint:
                prompt_list.append(ind_constraint['prompt'])
        final_prompt = polish_prompt(prompt_list)

        final_instruct_list.append({
            'id': cnt,
            'source': source,
            'content': content, 
            'gpt4o_polish_prompt': gpt4o_polish_prompt,
            'gpt4o_polish_response': gpt4o_polish_response,
            'info': info,
            'final_instruction': final_prompt,
            'constraint_info': constraint_ins
        })
        cnt += 1
    if cnt % 50 == 0:
        print(cnt)
        
with open('', 'w', encoding='utf-8') as file:
    for entry in final_instruct_list:
        file.write(json.dumps(entry) + '\n')


#### two

cnt = 0
final_instruct_list = []
for item in input_data:
    item_dict = json.loads(item)
    source = item_dict['source']
    content = item_dict['content']
    gpt4o_polish_prompt = item_dict['gpt4o_polish_prompt']
    gpt4o_polish_response = item_dict['gpt4o_polish_response']
    info = item_dict['info']
    constraint_func = random.sample(constraint_list, 2)
    task_instruct = "Polish the following text."
    # constraint_ins = constraint_func(item_dict)
    constraint_ins = []
    for sub_area in constraint_func:
        tmp_func = random.choice(sub_area)
        tmp_constraint = tmp_func(input_dict=item_dict)
        constraint_ins.append(tmp_constraint)

    if constraint_ins:
        prompt_list = [task_instruct]
        for ind_constraint in constraint_ins:
            if ind_constraint:
                prompt_list.append(ind_constraint['prompt'])
        final_prompt = polish_prompt(prompt_list)

        final_instruct_list.append({
            'id': cnt,
            'source': source,
            'content': content, 
            'gpt4o_polish_prompt': gpt4o_polish_prompt,
            'gpt4o_polish_response': gpt4o_polish_response,
            'info': info,
            'final_instruction': final_prompt,
            'constraint_info': constraint_ins
        })
        cnt += 1
    if cnt % 50 == 0:
        print(cnt)
        
with open('', 'w', encoding='utf-8') as file:
    for entry in final_instruct_list:
        file.write(json.dumps(entry) + '\n')

### one

cnt = 0
final_instruct_list = []
for item in input_data:
    item_dict = json.loads(item)
    source = item_dict['source']
    content = item_dict['content']
    gpt4o_polish_prompt = item_dict['gpt4o_polish_prompt']
    gpt4o_polish_response = item_dict['gpt4o_polish_response']
    info = item_dict['info']
    constraint_func = random.choice(constraint_list)
    task_instruct = "Polish the following text."
    # constraint_ins = constraint_func(item_dict)
    constraint_ins = []
    for sub_area in [constraint_func]:
        tmp_func = random.choice(sub_area)
        tmp_constraint = tmp_func(input_dict=item_dict)
        constraint_ins.append(tmp_constraint)

    if constraint_ins:
        prompt_list = [task_instruct]
        for ind_constraint in constraint_ins:
            if ind_constraint:
                prompt_list.append(ind_constraint['prompt'])
        final_prompt = polish_prompt(prompt_list)

        final_instruct_list.append({
            'id': cnt,
            'source': source,
            'content': content, 
            'gpt4o_polish_prompt': gpt4o_polish_prompt,
            'gpt4o_polish_response': gpt4o_polish_response,
            'info': info,
            'final_instruction': final_prompt,
            'constraint_info': constraint_ins
        })
        cnt += 1
    if cnt % 50 == 0:
        print(cnt)
        
with open('', 'w', encoding='utf-8') as file:
    for entry in final_instruct_list:
        file.write(json.dumps(entry) + '\n')