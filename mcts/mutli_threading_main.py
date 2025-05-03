import os
import sys
import json
import time
import random
import argparse
import logging
from vllm import LLM
from tqdm import tqdm
from planning.agent import Agent
from searching.search import TreeSearch
from reward_modeling.rm import RM
from utils.utils import read_data, read_wiki, save_one_result, load_prompt_template, save_results, read_jsonl_data_all
from transformers import AutoTokenizer
from models import PolicyModel
import threading

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = ""

def run_mcts(item, thread_results, thread_id, args, reward_model):
    idx = item["id"]
    test_input = item["test_input"]
    # docs = sample["docs"]
    ori_text = item["other_info"]["input_content"]
    
    tool_calls = item['tool_test_result']['constraint_info']
    tool_list = []
    for idx in range(len(tool_calls)):
        tool_list.append(tool_calls[idx]['function_call'])

    # tool_list = item['tool_test_result']['tool_list_response']['simpo_iter4']
    initial_state = [{'role': 'user', 'content': test_input}]
    try:
        agent = Agent(
            reward_model,
            initial_state,
            ppl_weight = args.ppl_weight,
            bart_weight = args.bart_weight,
            some_weight = args.some_weight,
            cons_weight = args.cons_weight,
            quality_weight = args.quality_weight,
            plan_api_key = args.plan_api_key,
            plan_temp = args.plan_temp,
            plan_url = args.plan_url,
            plan_model = args.plan_model,
            plan_presence_penalty = args.plan_presence_penalty,
            plan_frequency_penalty = args.plan_frequency_penalty,
            plan_n = args.plan_n,
            plan_top_p = args.plan_top_p,
            polish_api_key = args.polish_api_key,
            polish_temp = args.polish_temp,
            polish_url = args.polish_url,
            polish_model = args.polish_model,
            polish_presence_penalty = args.polish_presence_penalty,
            polish_frequency_penalty = args.polish_frequency_penalty,
            polish_n = args.polish_n,
            polish_top_p = args.polish_top_p,
        )

        # run MCTS
        mcts_search = TreeSearch(
            "MCTS",
            agent,
            initial_state,
            ori_text = ori_text,
            tool_list = tool_list,
            c_param=args.c_param,
            value_threshold=args.value_threshold,
            reflexion_threshold=args.reflexion_threshold,
        )

        logger.info("Run MCTS search for the {}-th sample".format(idx))
        start_time = time.time()
        # print("Test Input: " + test_input)
        best_mcts_action, search_tree = mcts_search.run_search(
            num_simulations=args.num_simulation,
            strategy="max_terminal",
            max_num_layers=args.max_num_layers,
            expand_probability=args.expand_probability
        )
        logger.info("Finish MCTS search for the {}-th sample in {}s".format(idx, time.time() - start_time))

        logger.info("Best MCTS Action with value {}:\n{}".format(best_mcts_action.value(), best_mcts_action.candidate_answer().strip()))
        ori_text, polished_text, tools_list, _, _ = best_mcts_action.get_trajectory()
        item["tool_list"] = tools_list
        item["polished_text"] = polished_text
        item["message_history"] = best_mcts_action.state
        logger.info(f"Finish the {idx}-th sample successfully")
        thread_results[thread_id] = item
    except:
        logger.exception(sys.exc_info())
        logger.info(f"Fail to complete the {idx}-th sample")
        thread_results[thread_id] = None


def main(args):
    logger.info("==== Hyper-Parameter Settings ====")
    logger.info(f"PPL weight: {args.ppl_weight}")
    logger.info(f"BART weight: {args.bart_weight}")
    logger.info(f"SOME weight: {args.some_weight}")
    logger.info(f"Constraint weight: {args.cons_weight}")
    logger.info(f"Quality weight: {args.quality_weight}")
    logger.info(f"Planner's model: {args.plan_model}")
    logger.info(f"Polish's model: {args.polish_model}")

    logger.info(f"Number of Simulations: {args.num_simulation}")
    logger.info(f"Max Number of Layers: {args.max_num_layers}")
    logger.info(f"Expand Probability: {args.expand_probability}")
    logger.info(f"Exploration Parameter: {args.c_param}")
    logger.info(f"Value Threshold: {args.value_threshold}")
    logger.info(f"Reflexion Threshold: {args.reflexion_threshold}")

    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Start Index: {args.start_idx}")
    logger.info(f"End Index: {args.end_idx}")

    logger.info(f"Log Path: {args.log_path}")
    logger.info(f"Save Path: {args.save_path}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"GPU Indices: {args.gpu_ids}")
    logger.info(f"Parallel Number: {args.parallel_num}")
    logger.info("============================")

    random.seed(args.seed)

    # load data
    # data = read_data(args.data_path, args.start_idx, args.end_idx)
    #### testing:
    # args.end_idx = 1
    data = read_jsonl_data_all(args.data_path, args.start_idx, args.end_idx)

    logger.info("Load {} samples from {}".format(len(data), args.data_path))

    # load reward model
    reward_model = RM()
    logger.info("Load reward models")

    # check completed samples
    completed_idx = []
    file_start = args.save_path.rfind("/")
    save_dir = args.save_path[:file_start]
    if os.path.exists(save_dir):
        for result_file in os.listdir(save_dir):
            ### jsonline
            finish_samples = [json.loads(i) for i in open(os.path.join(save_dir, result_file)).readlines()]
            for sam in finish_samples:
                completed_idx.append(sam["id"])

            ### json
            # with open(os.path.join(save_dir, result_file), "r") as f:
            #     finish_samples = json.load(f)
            #     for sam in finish_samples:
            #         completed_idx.append(sam["id"])
        logger.info("Load {} completed samples from {}".format(len(completed_idx), save_dir))
    else:
        os.makedirs(save_dir)

    left_data = []
    for sample in tqdm(data, desc="MCTS Running"):
        idx = sample["id"]
        if idx in completed_idx:
            continue
        left_data.append(sample)
    logger.info("Remaining data amount {}".format(len(left_data)))
    parallel_num = args.parallel_num

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
                t = threading.Thread(target=run_mcts, args=(item, thread_results, thread_id, args, reward_model))
                # time.sleep(60)
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
            with open(args.save_path, "a") as outfile:
                for entry in c_data_to_save:
                    json.dump(entry, outfile)
                    outfile.write("\n")
            logger.info(f'{len(c_data_to_save)} are generated ...')
        if len(error_items) == 0:
            break
        else:
            left_data = error_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # mcts setting
    # parser.add_argument("--policy_model_path", type=str, default="gpt-4o-2024-08-06")
    # parser.add_argument("--reference_model_path", type=str, default="Llama-3-8B-Instruct")
    # parser.add_argument("--reward_model_path", type=str, default="Llama-3-8B-SFR-Iterative-DPO-R")
    # parser.add_argument("--autoais_model_path", type=str, default="t5_xxl_true_nli_mixture")
    # parser.add_argument("--retriever", type=str, default="gtr", help="options: bm25/gtr")
    # parser.add_argument("--retrieval_topk", type=int, default=3, help="top-k documents for answering")
    # parser.add_argument("--query_sample", type=int, default=5, help="number of sampling search queries")
    # parser.add_argument("--answer_sample", type=int, default=1, help="number of sampling answers for each query")
    parser.add_argument("--num_simulation", type=int, default=30)

    # search setting
    parser.add_argument("--max_num_layers", type=int, default=4)
    parser.add_argument("--expand_probability", type=float, default=0.2)
    parser.add_argument("--c_param", type=float, default=0, help="co-efficient to control exploration")
    parser.add_argument("--value_threshold", type=float, default=0)
    parser.add_argument("--reflexion_threshold", type=int, default=10)
    
    # reward calculation setting
    parser.add_argument("--ppl_weight", type=float, default=0.33)
    parser.add_argument("--bart_weight", type=float, default=0.33)
    parser.add_argument("--some_weight", type=float, default=0.34)
    parser.add_argument("--cons_weight", type=float, default=0.5)
    parser.add_argument("--quality_weight", type=float, default=0.5)

    # LLL setting
    parser.add_argument("--plan_api_key", type=str, default="none")
    parser.add_argument("--plan_temp", type=float, default=0.7)
    parser.add_argument("--plan_url", type=str, default="http://0.0.0.0:6696/v1")
    parser.add_argument("--plan_model", type=str, default='/home/llm/models/llama3.1/temp_cache/agent_planner/iter4/checkpoint-80_merged')
    parser.add_argument("--plan_presence_penalty", type=float, default=0.0)
    parser.add_argument("--plan_frequency_penalty", type=float, default=0.0)

    parser.add_argument("--plan_n", type=int, default=3)
    parser.add_argument("--plan_top_p", type=int, default=1)

    parser.add_argument("--polish_api_key", type=str, default="none")
    parser.add_argument("--polish_temp", type=float, default=0.7)
    parser.add_argument("--polish_url", type=str, default="http://0.0.0.0:8000/v1")
    parser.add_argument("--polish_model", type=str, default='/home/llm/models/llama3.1/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--polish_presence_penalty", type=float, default=0.0)
    parser.add_argument("--polish_frequency_penalty", type=float, default=0.0)

    parser.add_argument("--polish_n", type=int, default=1)
    parser.add_argument("--polish_top_p", type=int, default=1)

    # dataset setting
    parser.add_argument("--data_path", type=str, default="agent_data/test.agent.jsonl")
    # parser.add_argument("--prompt_path", type=str, default="prompts/asqa.yaml")
    parser.add_argument("--start_idx", type=int, default=0, help="start index of samples")
    parser.add_argument("--end_idx", type=int, default=None, help="end index of samples")

    # general setting
    parser.add_argument("--log_path", type=str, default='logging/test.txt')
    parser.add_argument("--save_path", type=str, default='output/test.json')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ids", type=str, help="gpu indices")
    parser.add_argument("--parallel_num", type=int, default=10)

    args = parser.parse_args()

    # set gpus
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # create log directories if they don't exist
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    file_handler = logging.FileHandler(args.log_path, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    main(args)
