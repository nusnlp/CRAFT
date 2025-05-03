### ConsTRev:
1. generate the ConsTRev's constraint with: `put_constraint.py'

### TRIPS:

Build the planner:
1. create synthetic trajectory using GPT-4o with: `python create_syn_data_from_gpt4o.py'
2. build an initial planner:
    a. go to 'finetune/train_script' folder.
    b. run `bash run.sh' script.
3. build self-training data with: `python create_self_training_data.py'
4. enhance the planner via self-training:
    a. go to `align/runs' folder.
    b. run `bash llama3_instruct_cpo_simpo.sh' script.
5. iterate through step 4 and 5.

Run Search:
1. go to mcts folder
2. Pass in the path for the planner and reviser in the `run.sh' file
3. run `bash run.sh' script