#!/bin/bash
model_owner=$1
model=$2
lang=$3
turn=$4
user=$5
tensor_parallel_size=$6
role=$7


generate() {
    python tasks/dialogue_generation/generate_turn.py \
        --service vllm \
        --model ${model_owner}/${model} \
        --context ${context} \
        --role ${role} \
        --run_id ${run_id} \
        --type generate \
        --turn ${turn} \
        --lang ${lang}
}

process_batch() {
    local input_file=$1
    python agents/vllm_batch.py \
        --input_file ${input_file} \
        --tensor_parallel_size ${tensor_parallel_size}
}

upload() {
    local input_file=$1
    local provider=$2
    local key=$3
    python agents/gpt.py \
        --input_file ${input_file} \
        --type upload \
        --api_key ${key} \
        --provider ${provider}
}

evaluate() {
    python tasks/dialogue_generation/generate_turn.py \
        --service openai \
        --model ${model} \
        --context ${context} \
        --role user \
        --run_id ${run_id} \
        --type evaluate \
        --turn ${turn} \
        --lang ${lang}
}

process() {
    python tasks/dialogue_generation/generate_turn.py \
        --service vllm \
        --model ${model_owner}/${model} \
        --context ${context} \
        --role ${role} \
        --run_id ${run_id} \
        --type process \
        --turn ${turn} \
        --lang ${lang}
}

# Set up paths based on role
if [ "$role" = "user" ]; then
    context_turn=$((turn-1))
    context=dialogues/${lang}/${model}_${user}/${user}/turn-${context_turn}_${user}
    run_id=${model}_${user}
    output_dir=batches_to_process/${lang}/${model}_${user}/${model}
    dialogue_file=dialogues/${lang}/${model}_${user}/${user}/turn-${context_turn}_${user}
    input_batch=completed_batches/${lang}/${model}_${user}/${model}/turn-${turn}_${model}.jsonl
else
    context=dialogues/${lang}/${user}_${model}/${user}/turn-${turn}_${user}
    run_id=${user}_${model}
    output_dir=batches_to_process/${lang}/${user}_${model}/${model}
    dialogue_file=dialogues/${lang}/${user}_${model}/${user}/turn-${turn}_${user}
    input_batch=completed_batches/${lang}/${user}_${model}/${model}/turn-${turn}_${model}.jsonl
fi
# Generate initial responses
generate
process_batch ${output_dir}/turn-${turn}_${model}.jsonl
echo "Generated initial responses"

# Only validate user role responses
if [ "$role" = "user" ]; then
    # Evaluate responses
    evaluate
    upload ${output_dir}/turn-${turn}_${model}_eval.jsonl google $GEMENI_KEY
    echo "Completed evaluation"
    
    # Iterative generation until return 0
    process
    status=$?
    echo "Regenerating $status examples"
    
    iteration=0
    max_iterations=5
    
    while [ $status -ne 0 ] && [ $iteration -lt $max_iterations ]; do
        process_batch ${output_dir}/turn-${turn}_${model}_regen.jsonl
        evaluate
        upload ${output_dir}/turn-${turn}_${model}_eval.jsonl google $GEMENI_KEY
        process
        status=$?
        
        echo "Regenerating $status examples"
        iteration=$((iteration + 1))
    done
    
    if [ $iteration -ge $max_iterations ]; then
        echo "Warning: Reached maximum iterations without completion"
    fi
fi

# Process the final validated batch
python dialogues/process_batch.py \
    --role ${role} \
    --lang ${lang} \
    --model ${model} \
    --input_batch ${input_batch} \
    --dialogue_file ${dialogue_file}
