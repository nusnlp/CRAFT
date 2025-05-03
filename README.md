### Install Packages:
`pip install -r requirements.txt`

### ConsTRev Dataset Creation Pipeline:
Generate the ConsTRev's constraint instruction with: `put_constraint.py`

### TRIPS:

Build the planner:

1. Create synthetic trajectory using GPT-4o with: `python create_syn_data_from_gpt4o.py`
2. Build an initial planner:
    a. Go to `finetune/train_script` folder.
    b. Run `bash run.sh` script.
3. Build self-training data with: `python create_self_training_data.py`
4. Enhance the planner via self-training:
    a. Go to `align/runs` folder.
    b. Run `bash llama3_instruct_cpo_simpo.sh` script.
5. Iterate through steps 4 and 5.

Run Search:
1. Go to the MCTS folder
2. Pass in the path for the planner and reviser in the `run.sh` file
3. run `bash run.sh` script

### Test Data and Predictions:
