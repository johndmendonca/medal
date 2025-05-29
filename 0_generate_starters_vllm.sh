model_owner=$1
model=$2
lang=$3
tensor_parallel_size=$4
original_dataset=$5

run_id=affective_persona
dataset_path=tasks/narrative_generation/data/${original_dataset}.json
full_model="${model_owner}/${model}"

generate() {
    python tasks/narrative_generation/generate_narratives.py \
        --lang ${lang} \
        --model ${full_model} \
        --type generate \
        --run-id ${run_id} \
        --dataset ${dataset_path}
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
    python tasks/narrative_generation/generate_narratives.py \
        --lang ${lang} \
        --model ${model} \
        --type evaluate \
        --run-id ${run_id} \
        --dataset ${dataset_path}
}

process() {
    python tasks/narrative_generation/generate_narratives.py \
        --lang ${lang} \
        --model ${full_model} \
        --type process \
        --run-id ${run_id} \
        --dataset ${dataset_path}
}

# First generation
generate
process_batch batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}.jsonl
echo "Retrieved generations"

evaluate
upload batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}_eval.jsonl google ${GEMENI_KEY}
echo "Retrieved evaluations"

# Iterative generation until return 0
process
status=$?

echo "Retrieved processed generations."
echo "Regenerating $status examples."

iteration=0
max_iterations=10  # Safety limit to avoid infinite loops

while [ $status -ne 0 ] && [ $iteration -lt $max_iterations ]; do
    process_batch batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}_regen.jsonl
    evaluate
    upload batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}_eval.jsonl google ${GEMENI_KEY}
    process
    status=$?
    
    echo "Regenerating $status examples."
    iteration=$((iteration + 1))
done

if [ $iteration -ge $max_iterations ]; then
    echo "Warning: Reached maximum iterations without completion."
fi

python dialogues/process_batch.py \
    --role user \
    --lang ${lang} \
    --model ${model} \
    --input_batch completed_batches/batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}.jsonl \
    --source batches_to_process/${run_id}/${original_dataset}_${lang}_turn0_${model}.jsonl