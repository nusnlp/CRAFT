#!/bin/bash

query_sample="3"
answer_sample="1"
retrieval_topk="3"
num_simulation="30"
max_num_layers="5"
expand_probability="0.4"
c_param="0.2"
# value_threshold="1.0"
value_threshold=85
seed="42"
reflexion_threshold="10"
parallel_num=50

cons_type=tetra_ana

data_path=test.agent.${cons_type}.jsonl
# 0-100
start_idx=0
end_idx="100"
# end_idx=128
log_path=test.agent.${cons_type}.${start_idx}-${end_idx}.txt
save_path=test.agent.${cons_type}.${start_idx}-${end_idx}.json


python3 mutli_threading_main.py --num_simulation ${num_simulation} \
              --max_num_layers ${max_num_layers} \
              --expand_probability ${expand_probability} \
              --c_param ${c_param} \
              --value_threshold ${value_threshold} \
              --data_path ${data_path} \
              --start_idx ${start_idx} \
              --end_idx ${end_idx} \
              --log_path ${log_path} \
              --save_path ${save_path} \
              --seed ${seed} \
              --parallel_num ${parallel_num}