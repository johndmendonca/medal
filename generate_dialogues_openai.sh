#!/bin/bash
user=$1
model_owner=$2
model=$3
tensor_parallel_size=$4
batched=${5:-False}

original_dataset=ATOMIC10X_persona_1k_3

langs=(chinese english french german portuguese spanish)

for lang in "${langs[@]}"; do
    echo "Processing language: ${lang}"
    
    # Generate and validate turn 0 user starters
    echo "Generating validated user starters for ${lang}"
    ./0_generate_starters_openai.sh ${user} ${lang} ${original_dataset}
    
    # Generate turn 0 assistant responses
    echo "Generating turn 0 assistant responses for ${lang}"
    ./1_generate_turn0_vllm.sh ${model_owner} ${model} ${lang} ${user} ${tensor_parallel_size} ${original_dataset}
    
    # Generate subsequent turns with validation for user responses
    for turn in {0..3}; do
        # Generate and validate user responses
        echo "Generating turn $((turn+1)) user responses for ${lang} with validation"
        ./2_upload_turnX_openAI.sh ${user} ${turn} ${model} ${lang} ${batched}
        
        ./3_download_turnX_openAI.sh ${user} $((turn+1)) ${model} ${lang} ${batched}
        
        # Generate assistant responses (no validation needed)
        echo "Generating turn $((turn+1)) assistant responses for ${lang}"
        ./4_generate_turnX.sh ${model_owner} ${model} ${lang} $((turn+1)) ${user} ${tensor_parallel_size} assistant
    done
done

echo "All dialogue generation completed successfully."