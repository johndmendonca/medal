#!/bin/bash
user=$1
turn=$2
model=$3
lang=$4
batched=${5:-False}

output_dir=batches_to_process/${lang}/${user}_${model}/${user}
run_context=dialogues/${lang}/${user}_${model}/${model}/turn-$((turn-1))_${model}

download() {
    python agents/gpt.py \
        --batch_id submitted_batches/${lang}/${user}_${model}/${user}/turn-${turn}_${user}.json \
        --type download \
        --api_key ${OPENAI_KEY}
}

upload() {
    local input_file=$1
    python agents/gpt.py \
        --input_file ${input_file} \
        --type upload \
        --api_key ${OPENAI_KEY} \
        --batched ${batched}
}

evaluate() {
    python tasks/dialogue_generation/generate_turn.py \
        --service openai \
        --model ${user} \
        --context ${run_context} \
        --role user \
        --run_id ${user}_${model} \
        --type evaluate \
        --turn ${turn} \
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
        --turn ${turn} \
        --lang ${lang}
}

# Download the batch if we're in batched mode
if [ "$batched" = "True" ]; then
    download
    echo "Downloaded initial responses"
    
    # Run validation cycle for batched processing
    evaluate
    upload ${output_dir}/turn-${turn}_${user}_eval.jsonl
    echo "Uploaded evaluation requests"
    
    download
    echo "Downloaded evaluations"
    
    process
    status=$?
    echo "Regenerating $status examples"
    
    iteration=0
    max_iterations=5
    
    while [ $status -ne 0 ] && [ $iteration -lt $max_iterations ]; do
        upload ${output_dir}/turn-${turn}_${user}_regen.jsonl
        download
        
        evaluate
        upload ${output_dir}/turn-${turn}_${user}_eval.jsonl
        download
        
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
    --role user \
    --lang ${lang} \
    --model ${user} \
    --input_batch completed_batches/${lang}/${user}_${model}/${user}/turn-${turn}_${user}.jsonl \
    --dialogue_file ${run_context}
