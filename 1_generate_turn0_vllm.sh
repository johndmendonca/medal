#!/bin/bash

#generate the batch
model_owner=$1
model=$2
lang=$3
user=$4
tensor_parallel_size=$5
original_dataset=$6
run_id=affective_persona

python tasks/dialogue_generation/generate_turn.py \
    --service vllm \
    --model ${model_owner}/${model} \
    --context dialogues/completed_batches/batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${user} \
    --role assistant \
    --run_id ${user}_${model} \
    --turn 0 \
    --lang ${lang} \
    --type generate

#run the batch
 python agents/vllm_batch.py \
    --input_file batches_to_process/${lang}/${user}_${model}/${model}/turn-0_${model}.jsonl \
    --tensor_parallel_size ${tensor_parallel_size}

#process the batch
python dialogues/process_batch.py \
    --role assistant \
    --lang ${lang} \
    --model ${model} \
    --input_batch completed_batches/${lang}/${user}_${model}/${model}/turn-0_${model}.jsonl \
    --dialogue_file dialogues/completed_batches/batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${user}

