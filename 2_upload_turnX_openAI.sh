#!/bin/bash
user=$1
turn=$2
model=$3
lang=$4
batched=${5:-False}

next_turn=$((turn+1))
output_dir=batches_to_process/${lang}/${user}_${model}/${user}
run_context=dialogues/${lang}/${user}_${model}/${model}/turn-${turn}_${model}

generate() {
    python tasks/dialogue_generation/generate_turn.py \
        --service openai \
        --model ${user} \
        --context ${run_context} \
        --role user \
        --run_id ${user}_${model} \
        --type generate \
        --turn ${next_turn} \
        --lang ${lang}
}

upload() {
    local input_file=$1
    local provider=$2
    local key=$3
    python agents/gpt.py \
        --input_file ${input_file} \
        --type upload \
        --api_key ${key} \
        --batched ${batched} \
        --provider ${provider}
}

evaluate() {
    python tasks/dialogue_generation/generate_turn.py \
        --service openai \
        --model ${user} \
        --context ${run_context} \
        --role user \
        --run_id ${user}_${model} \
        --type evaluate \
        --turn ${next_turn} \
        --lang ${lang}
}

process() {
    python tasks/dialogue_generation/generate_turn.py \
        --service openai \
        --model ${user} \
        --context ${run_context} \
        --role user \
        --run_id ${user}_${model} \
        --type process \
        --turn ${next_turn} \
        --lang ${lang}
}

# First generation
generate

if [ "$batched" = "False" ]; then
    # If not batched, run the full validation cycle immediately
    upload ${output_dir}/turn-${next_turn}_${user}.jsonl openai $OPENAI_KEY
    
    # First evaluation
    evaluate
    upload ${output_dir}/turn-${next_turn}_${user}_eval.jsonl google $GEMENI_KEY
    
    # Iterative generation until return 0
    process
    status=$?
    #echo "Regenerating $status examples"
    
    iteration=0
    max_iterations=5
    
    while [ $status -ne 0 ] && [ $iteration -lt $max_iterations ]; do
        upload ${output_dir}/turn-${next_turn}_${user}_regen.jsonl openai $OPENAI_KEY
        evaluate
        upload ${output_dir}/turn-${next_turn}_${user}_eval.jsonl google $GEMENI_KEY
        process
        status=$?
        
        #echo "Regenerating $status examples"
        iteration=$((iteration + 1))
    done
    
    if [ $iteration -ge $max_iterations ]; then
        echo "Warning: Reached maximum iterations without completion"
    fi

    
else
    # If batched, just upload the initial generation
    upload ${output_dir}/turn-${next_turn}_${user}.jsonl openai $OPENAI_KEY
    echo "Uploaded initial responses for batch processing"
fi