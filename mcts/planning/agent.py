import os
import pdb
import time
import torch
import random
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from searching.tree import Node
import copy
from openai import OpenAI
import re 
import nltk
from gliner import GLiNER

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

class Agent:
    def __init__(
            self,
            reward_model,
            initial_state,
            ppl_weight = 0.33,
            bart_weight = 0.33,
            some_weight = 0.34,
            cons_weight = 0.5,
            quality_weight = 0.5,
            plan_api_key = "none",
            plan_temp = 0.7,
            plan_url = "",
            plan_model = '',
            plan_presence_penalty = 0,
            plan_frequency_penalty = 0,
            plan_n = 3,
            plan_top_p = 1,
            polish_api_key = "none",
            polish_temp = 0.7,
            polish_url = "",
            polish_model = '',
            polish_presence_penalty = 0,
            polish_frequency_penalty = 0,
            polish_n = 1,
            polish_top_p = 1,
    ):
        ### 需要定义计算reward的方式，policy model（LLama3 / GPT-4o）， planner，以及各种tools，feedback也要放进去
        self.reward_model = reward_model

        self.initial_state = initial_state

        ### reward parameter
        self.ppl_weight = ppl_weight
        self.bart_weight = bart_weight
        self.some_weight = some_weight
        self.cons_weight = cons_weight
        self.quality_weight = quality_weight

        ### planner's parameter
        self.plan_api_key = plan_api_key
        self.plan_temperature = plan_temp
        self.plan_url = plan_url
        self.plan_model = plan_model
        self.plan_presence_penalty = plan_presence_penalty
        self.plan_frequency_penalty = plan_frequency_penalty
        self.plan_n = plan_n
        self.plan_top_p = plan_top_p

        ### polish's parameter
        self.polish_api_key = polish_api_key
        self.polish_temperature = polish_temp
        self.polish_url = polish_url
        self.polish_model = polish_model
        self.polish_presence_penalty = polish_presence_penalty
        self.polish_frequency_penalty = polish_frequency_penalty
        self.polish_n = polish_n
        self.polish_top_p = polish_top_p

        ### ner model
        self.ner_model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")

        ### feedback model's parameter
        self.feedback_api_key = polish_api_key
        self.feedback_temp = polish_temp
        self.feedback_base_url = polish_url
        self.feedback_base_model = polish_model

    def get_plan(self, tmp_message):
        api_key = self.plan_api_key
        api_base = self.plan_url
        MODEL_NAME = self.plan_model
        client = OpenAI(api_key=api_key, base_url=api_base)
        outputs = []
        for i in range(self.plan_n):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=tmp_message,
                temperature=self.plan_temperature,
                presence_penalty=self.plan_presence_penalty,
                frequency_penalty=self.plan_frequency_penalty,
                )
            outputs.append(response.choices[0].message.content)
        # response = client.chat.completions.create(
        #     model=MODEL_NAME,
        #     messages=tmp_message,
        #     temperature=self.plan_temperature,
        #     presence_penalty=self.plan_presence_penalty,
        #     frequency_penalty=self.plan_frequency_penalty,
        #     n = self.plan_n,
        #     top_p = self.plan_top_p
        # )
        # outputs = [response.choices[i].message.content for i in range(len(response.choices))]
        # return response.choices[0].message.content
        return outputs
    
    def process_tool_lists(self, tool_list):
        final_tool_list = []
        for tool in tool_list:
            if 'python' in tool:
                continue
            final_tool_list.append(tool)
        if 'text_eval()' not in final_tool_list:
            final_tool_list.append('text_eval()')
        return final_tool_list

    def get_action_and_tool(self, response):
        thought_pattern = r"###\s*THOUGHT:\n(.*?)\n###"
        tools_pattern = r"###\s*TOOLS:\n(.*?)\n###"
        plan_pattern = r"###\s*PLAN:\n(.*)"
        tools_list_pattern = r"\d+\.\s`([^`]+)`"
        tools_list = []
        thought = ""
        thought_match = re.search(thought_pattern, response, re.DOTALL)
        tools_match = re.search(tools_pattern, response, re.DOTALL)
        plan_match = re.search(plan_pattern, response, re.DOTALL)
        plan = ""
        if thought_match:
            thought = thought_match.group(1).strip()
        if tools_match:
            tools = tools_match.group(1).strip().replace('```', '').replace("```", "")
            tools_list = [tool.strip() for tool in tools.splitlines() if tool.strip()]
            final_tool_list = self.process_tool_lists(tools_list)
            # if 'text_eval()' not in tools_list:
            #     tools_list.append('text_eval()')
        if plan_match:
            plan = plan_match.group(1).strip()
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(response)
        # print("Thought: {}".format(thought))
        # print("Tools: {}".format(tools))
        # print("Plan: {}".format(plan))
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(final_tool_list)
        return [], [], plan
    
    def format_list(self, items):
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

    def remove_last_period(self, input_string):
        last_period_index = input_string.rfind('.')
        if last_period_index == -1:
            return input_string
        else:
            return input_string[:last_period_index] + input_string[last_period_index+1:]

    def add_keyword_constraint(self, text):
        pronouns = {"i", "me", "my", "mine", "myself",
                "we", "us", "our", "ours", "ourselves",
                "you", "your", "yours", "yourself", "yourselves",
                "he", "him", "his", "himself",
                "she", "her", "hers", "herself",
                "it", "its", "itself",
                "they", "them", "their", "theirs", "themselves"}
        if '\nImprovement' in text or '\n Improvement' in text:
            pattern = r'Sentence:\s*(.*?)\nImprovement Plan:\s*(.*?)(?=\nSentence:|\Z)'
            matches = re.findall(pattern, text, re.DOTALL)
            if not matches:
                pattern = r'''
                    Sentence\s*\d+:               # Match 'Sentence' followed by optional spaces and digits
                    \s*                           # Optional whitespace
                    (?:"|“)?                      # Optional opening quote (" or “)
                    (.*?)                         # Capture the sentence content
                    (?:"|”)?                      # Optional closing quote (" or ”)
                    \s*                           # Optional whitespace
                    Improvement\ Plan:\s*         # Match 'Improvement Plan:' with optional spaces
                    (.*?)                         # Capture the improvement plan
                    (?=\n?Sentence\s*\d+:|$)      # Lookahead for the next 'Sentence n:' or end of string
                '''
                matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.VERBOSE)
        else:
            # print("no constrains example")
            pattern = r'Sentence:\s*"(.*?)";\s*Improvement Plan:\s*(.*?)(?=Sentence:|$)'
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if not matches:
                pattern = r'Sentence:\s*(.*?);\s*Improvement Plan:\s*(.*?)(?=Sentence:|$)'
                matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
                if not matches:
                    pattern = r'^\s*\d+\.\s*Sentence:\s*"(?P<sentence>.*?)"\s*\n\s*Improvement Plan:\s*(?P<plan>.*?)(?=(^\s*\d+\.\s|$))'
                    matches_iter = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
                    matches = [(m.group('sentence'), m.group('plan')) for m in matches_iter]
                    if not matches:
                        pattern = r'Sentence:\s*(.*?)\s*Improvement Plan:\s*(.*?)(?=Sentence:|$)'
                        matches = re.findall(pattern, text, re.DOTALL)
        modified_entries = []
        for sentence, plan in matches:
            keywords = []
            token_lists = nltk.word_tokenize(sentence)
            labels = ["Person", "Award", "Date", "Competitions", "Company"]
            entities = self.ner_model.predict_entities(sentence, labels, threshold=0.5)
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
                constraint = self.format_list(keywords)
                tmp_plan = self.remove_last_period(plan)
                new_plan = tmp_plan + ", keep the keywords " + constraint + "."
            else:
                new_plan = plan
            modified_entry = f'Sentence: "{sentence}"; Improvement Plan: "{new_plan}"'
            modified_entries.append(modified_entry)
        modified_text = '\n'.join(modified_entries)
        return modified_text

    def generate_final_prompt(self, input):
        editing_plan = input.split('###PLAN:')[-1].replace("\n\n", "\n").replace("*", "").lstrip('\n')
        tmp_cleaned_text = re.sub(r'^\s+(Sentence:)', r'\1', editing_plan, flags=re.MULTILINE)
        cleaned_text = self.add_keyword_constraint(tmp_cleaned_text)
        ending_prompt = """### INSTRUCTIONS:

    Using the information provided in each text editing plan (### INPUT), generate the polished version of each sentence by applying the specified improvements. Maintain the original order of sentences.

    **In your output, provide only the final polished sentences, one after another, without any prefixes, numbering, or additional text.**

    ### INPUT:\n\n"""

        final_prompt = ending_prompt + cleaned_text + "\n\n### OUTPUT:\n\n"
        return final_prompt

    def get_gpt4_polish_response(self, user_input):
        api_key = self.polish_api_key
        api_base = self.polish_url
        MODEL_NAME= self.polish_model
        client = OpenAI(api_key=api_key, base_url=api_base)
        polish_prompt = "You are a helpful writing assistant tasked with refining text."
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': polish_prompt},
                {"role": "user", "content": user_input},
                ],
            temperature=self.polish_temperature,
            presence_penalty=0,
            frequency_penalty=0,
            )
        return response.choices[0].message.content
    
    def convert_to_paragraph(self, text):
        lines = text.splitlines()
        lines = [line.strip() for line in lines if line.strip() != '']
        paragraph = ' '.join(lines)
        return paragraph

    def format_feedback(self, input_text):
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

    def get_coherence_feedback_response(self, user_input):
        api_key = self.feedback_api_key
        api_base = self.feedback_base_url
        MODEL_NAME = self.feedback_base_model
        client = OpenAI(api_key=api_key, base_url=api_base)
        coherence_check_prompt = "You are a helpful assistant tht helps to identify potential fluency problems in a given text."
        coherence_pre_prompt = "Please analyze the following text for coherence problems, such as unclear connections between ideas, lack of logical flow, abrupt transitions, or inconsistencies in the overall message. For each sentence or section that contains a coherence problem, format the output strictly as follows: 'Original: [original text]; Suggestion: [corrected text]'. If a sentence or section has no issues, do not include it in the output. Do not include any additional content. Text: "
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': coherence_check_prompt},
                {"role": "user", "content": coherence_pre_prompt + user_input},
                ],
            temperature=self.feedback_temp,
            presence_penalty=0,
            frequency_penalty=0,
            )
        return response.choices[0].message.content

    def get_fluency_feedback_response(self, user_input):
        api_key = self.feedback_api_key
        api_base = self.feedback_base_url
        MODEL_NAME = self.feedback_base_model
        client = OpenAI(api_key=api_key, base_url=api_base)
        fluency_check_prompt = "You are a helpful assistant tht helps to identify potential fluency problems in a given text."
        fluency_pre_prompt = "Please analyze the following text for fluency issues, including awkward phrasing, unnatural word choices, sentence flow, and readability problems. For each of the sentence that contain fluency problems, please format the output strictly as follows: 'Original: [original text]; Suggestion: [corrected text]'. If a sentence has no issues, do not include it in the output. Do not include any additional content. Text: "
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': fluency_check_prompt},
                {"role": "user", "content": fluency_pre_prompt + user_input},
                ],
            temperature=self.feedback_temp,
            presence_penalty=0,
            frequency_penalty=0,
            )
        return response.choices[0].message.content

    def get_gram_feedback_response(self, user_input):
        api_key = self.feedback_api_key
        api_base = self.feedback_base_url
        MODEL_NAME = self.feedback_base_model
        client = OpenAI(api_key=api_key, base_url=api_base)
        grammar_check_prompt = "You are a helpful assistant tht helps to identify potential grammatical errors in a given text."
        grammar_pre_prompt = "Please analyze the following text for grammatical errors, including issues with sentence structure, punctuation, subject-verb agreement, tense consistency, pronoun usage, and any other common grammar mistakes. For each of the sentence that contain grammar errors, please format the output strictly as follows: 'Original: [original text]; Suggestion: [corrected text]'. Do not include any additional content. Text: "
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': grammar_check_prompt},
                {"role": "user", "content": grammar_pre_prompt + user_input},
                ],
            temperature=self.feedback_temp,
            presence_penalty=0,
            frequency_penalty=0,
            )
        return response.choices[0].message.content

    def create_feedback_prompt(self, flu_fb, coh_fb, gra_fb):
        all_gra_fb = "\n".join(gra_fb)
        all_flu_fb = "\n".join(flu_fb)
        all_coh_fb = "\n".join(coh_fb)
        return "Text quality feedback:\nGrammatical Feedback: {}\n\nCoherence Feedback: {}\n\nFluency Feedback: Original: {}".format(all_gra_fb, all_coh_fb, all_flu_fb)

    def tool_fb_prompt(self, tools_list):
        tools_string = "; ".join(f"`{tool}`" for tool in tools_list)
        feedback = f"Tool usage feedback:\nThe tools are used correctly please keep using it: {tools_string}"
        return feedback

    def ob_prompt(self, polished_para, tool_acc_fb, text_quality_fb, text_cons_fb):
        prompt = "###OBSERVATION: The polished text:\n{}\n\nHere are the feedback:\n{}\n\n{}\n\n{}\n\nPlease use these feedback to further refine the text by generating new text refinement plans and tool usages according to the response format.".format(polished_para, tool_acc_fb, text_quality_fb, text_cons_fb)
        return prompt

    def get_init_ppl_bart(self, ori_text):
        init_ppl, init_bart = self.reward_model.get_ppl_bart(ori_text)
        # print("Initial PPL: {}".format(init_ppl))
        # print("Initial BART: {}".format(init_bart))
        return init_ppl, init_bart
    
    ### expansion 只call 这个function
    def perform_querying(self, simu_iter, leaf_node, leaf_node_layer, root_ppl, root_bart, root_tool_list):
        if leaf_node.is_terminated():
            return []
        ## state 存之前的message history
        leaf_node_state = leaf_node.state
        ori_depth = leaf_node.get_depth()
        ## get a number of LLMs' response
        new_message = copy.deepcopy(leaf_node_state)

        # print("========================")
        # print(leaf_node)
        # print(leaf_node.state)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(new_message)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(type(new_message))
        # # a = "a"
        # # print(a+1)

        thought_actions = self.get_plan(new_message)

        # print(thought_actions)
        querying_results = []
        print("-" * 10 + "Answering" + "-" * 10)
        for i, thought_action in enumerate(thought_actions):
            ori_ppl_list, ori_bart_list, ori_some_list, ori_cons_list = leaf_node.get_history_metrics()
            ppl_list = copy.deepcopy(ori_ppl_list)
            bart_list = copy.deepcopy(ori_bart_list)
            some_list = copy.deepcopy(ori_some_list)
            cons_list = copy.deepcopy(ori_cons_list)
            ### get polishment plan
            # thought, tools_list, plan = self.get_action_and_tool(thought_action)
            try:
                _, _, plan = self.get_action_and_tool(thought_action)
            except:
                continue
            if plan == "":
                # print("Plan is empty")
                continue
            new_message.append({'role': 'assistant', 'content': thought_action})
            # print(f"Node id: {simu_iter}_{i + 1}")
            # print(f"Search query {i + 1}: {plan.strip()}")
            # print(f"Search query {i + 1}:")
            ### get polished text
            polished_text = self.get_gpt4_polish_response(self.generate_final_prompt(plan))
            polished_paragraph = self.convert_to_paragraph(polished_text)
            ### get the tool, and text quality checking 

            ### create new node
            new_node = Node(id=f"{simu_iter}_{i + 1}", parent=leaf_node, state=new_message)
            new_node.set_candidate_answer(polished_paragraph)
            new_node.set_ori_text(leaf_node.ori_text)
            # new_node.set_tool_lists(tool_lists = tools_list)
            new_node.set_tool_lists(tool_lists = root_tool_list)
            new_node.set_depth(depth=ori_depth + 1)

            new_node.set_init_ppl_bart(init_ppl = root_ppl, init_bart = root_bart)
            ppl, some, bart, cons_acc, ini_ppl, ini_bart, text_cons_fb = self.reward_model.get_rollout_reward(new_node)
            # print("PPL: {}".format(ppl))
            # print("SOME: {}".format(some))
            # print("BART: {}".format(bart))
            # print("Cons ACC: {}".format(cons_acc))
            # print("Init PPL: {}".format(ini_ppl))
            # print("Init BART: {}".format(ini_bart))
            # print("Text Cons FB: {}".format(text_cons_fb))

            reward, normalized_ppl, normalized_bart, normalized_some = self.combined_score(ppl=ppl,
                                                                                           some=some, 
                                                                                           bart=bart, 
                                                                                           cons_acc=cons_acc, 
                                                                                           ini_ppl=ini_ppl, 
                                                                                           ini_bart=ini_bart)

            # print(f"normalized PPL: {normalized_ppl}, \nnormalized BART: {normalized_bart}, \nnormalized SOME: {normalized_some}, \nOverall Reward: {reward}")

            ### get the feedback
            fluency_feedback = self.format_feedback(self.get_fluency_feedback_response(polished_paragraph))
            coherence_feedback = self.format_feedback(self.get_coherence_feedback_response(polished_paragraph))
            grammar_feedback = self.format_feedback(self.get_gram_feedback_response(polished_paragraph))

            text_quality_fb = self.create_feedback_prompt(fluency_feedback, coherence_feedback, grammar_feedback)
            tool_acc_fb = self.tool_fb_prompt(new_node.get_tool_lists())
            observation = self.ob_prompt(polished_paragraph, tool_acc_fb, text_quality_fb, text_cons_fb)
            new_message.append({'role': 'user', 'content': observation})

            new_node.update_state(new_message)
            new_node._initial_value = reward
            querying_results.append((new_node, reward))
            
            ppl_list.append(ppl)
            bart_list.append(bart)
            some_list.append(some)
            cons_list.append(cons_acc)

            tmp_ppl_list, tmp_bart_list, tmp_some_list, tmp_cons_list = leaf_node.get_history_metrics()

            new_node.set_history_metrics(ppl_list=ppl_list, bart_list=bart_list, some_list=some_list, cons_list=cons_list)

            tmp_ppl_list, tmp_bart_list, tmp_some_list, tmp_cons_list = leaf_node.get_history_metrics()
            tmp_ppl_list, tmp_bart_list, tmp_some_list, tmp_cons_list = new_node.get_history_metrics()
            early_stop = None
            # if len(cons_list) > 2:
            #     target_node = new_node.parent()
            #     target_node.set_terminated()
            
            # if cons_acc > 0.8 and ori_depth >= 2:
            #     early_stop = self.early_stop(ppl_list, bart_list, some_list, patience=3)
            # if early_stop:
            #     target_node = new_node
            #     diff = len(cons_list) - early_stop
            #     for i in range(diff):
            #         if i == 0:
            #             target_node = new_node.parent()
            #         else:
            #             target_node = target_node.parent()
            #     print("Perform early stopping!!!")
            #     print("Node: {}, Early Stop value: {}, length of the list: {}, depth: {}".format(new_node.get_node_id(), early_stop, len(cons_list), new_node.get_depth()))
            #     print("Early stop diff: {}".format(diff))
            #     print("Current node is: {}".format(new_node.get_node_id()))
            #     if diff > 0:
            #         print("Target node is: {}".format(target_node.get_node_id()))
            #     print("=====================================")
            #     target_node.set_terminated()

        return querying_results
    
    def check_cons(self, cons_list):
        if len(set(cons_list)) == 1:
            return None
        max_element = max(cons_list)
        indices = [i+1 for i, x in enumerate(cons_list) if x == max_element]
        return indices

    def early_stop(self, ppl_list, bart_list, some_list, patience=2):
        ppl_list[0] = 10000000
        bart_list[0] = -1000
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
                return best_epoch + 1
        return None
    
    def combined_score(self, ppl, some, bart, cons_acc, ini_ppl, ini_bart, max_ppl = 0, min_bart = -10):
        ### normalize ppl
        min_ppl = ini_ppl
        # print(min_ppl)
        # print(max_ppl)
        # print(ppl)
        normal_ppl = 100 * (min_ppl - ppl) / (min_ppl - max_ppl)
        ### normalize bart
        max_bart = ini_bart
        normal_bart = 100 * (bart - min_bart) / (max_bart - min_bart)
        ### normalize some
        normal_some = 100 * some
        text_quality = self.ppl_weight * normal_ppl + self.bart_weight * normal_bart + self.some_weight * normal_some
        combined_score = self.cons_weight * 100 * cons_acc + self.quality_weight * text_quality
        return combined_score, normal_ppl, normal_bart, normal_some