#!/bin/bash
model_owner=$1
model=$2
tensor_parallel_size=$3
original_dataset=ATOMIC10X_persona_1k_3

langs=(chinese english french german portuguese spanish)

user_owner=google
user=gemma-3-27b-it

for lang in "${langs[@]}"; do
    # Generate and validate turn 0 user starters
    ./0_generate_starters_vllm.sh ${user_owner} ${user} ${lang} ${tensor_parallel_size} ${original_dataset}
    
    # Generate turn 0 assistant responses
    ./1_generate_turn0_vllm.sh ${model_owner} ${model} ${lang} ${user} ${tensor_parallel_size} ${original_dataset}
    
    # Generate subsequent turns with validation for user responses
    for turn in {1..4}; do
        echo "Generating turn ${turn} for ${lang} - user role with validation"
        # Generate and validate user responses
        ./4_generate_turnX.sh ${user_owner} ${user} ${lang} ${turn} ${model} ${tensor_parallel_size} user

        echo "Generating turn ${turn} for ${lang} - assistant role"
        # Generate assistant responses (no validation needed)
        ./4_generate_turnX.sh ${model_owner} ${model} ${lang} ${turn} ${user} ${tensor_parallel_size} assistant
    done
done
