"""
Dialogue generation module for MEDAL framework.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from medal.config import Config, ModelConfig
from medal.logging_config import get_logger
from medal.prompts import USER_PROMPT, CHATBOT_PROMPT, EVALUATE_PROMPT
from medal.utils import (
    load_jsonl,
    save_jsonl,
    load_dataset,
    build_dialogue_batch_path,
)

logger = get_logger(__name__)


class DialogueGenerator:
    """
    Generator for dialogue turns with validation and regeneration support.
    
    This class handles the generation, evaluation, and regeneration of dialogue
    turns for both user and assistant roles.
    """
    
    def __init__(
        self,
        context: str,
        lang: str,
        model: str,
        role: str,
        turn: int,
        run_id: str = "vanilla",
        config: Optional[Config] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        """
        Initialize the dialogue generator.
        
        Args:
            context: Path to dialogue context dataset
            lang: Language code
            model: Model identifier (e.g., "meta-llama/Llama-3.3-70B-Instruct")
            role: Role to generate ('user' or 'assistant')
            turn: Turn number
            run_id: Run identifier
            config: MEDAL configuration
            model_config: Model-specific configuration
        """
        self.context_path = context
        self.lang = lang
        self.model = model
        self.role = role
        self.turn = turn
        self.run_id = run_id
        self.config = config or Config()
        self.model_config = model_config or self.config.model
        
        # Set system prompt based on role
        self.sys_prompt = USER_PROMPT if role == 'user' else CHATBOT_PROMPT
        
        # Build file paths (dialogue uses lang/run_id/model structure)
        base_path = build_dialogue_batch_path(
            lang=lang,
            run_id=run_id,
            model_name=model,
            turn=turn,
            file_type="gen",
            base_dir=self.config.paths.batches_to_process,
        )
        self.gen_file_path = str(base_path)
        self.eval_file_path = str(build_dialogue_batch_path(lang=lang, run_id=run_id, model_name=model, turn=turn, file_type="eval", base_dir=self.config.paths.batches_to_process))
        self.regen_file_path = str(build_dialogue_batch_path(lang=lang, run_id=run_id, model_name=model, turn=turn, file_type="regen", base_dir=self.config.paths.batches_to_process))
        
        # Load dialogue context
        self.data = load_dataset(context)
    
    def generate(self) -> None:
        """
        Generate dialogue responses for each data input and write them to a file.
        
        This method generates responses for ongoing dialogues (those not marked as ended),
        creates appropriate prompts based on the role (user or assistant), and writes
        the requests to the output file.
        """
        requests = []
        
        for current_idx, data_input in enumerate(self.data):
            if data_input.get('ended', False):
                continue
            
            # Build custom ID
            model_base = self.model.split("/")[-1] if "/" in self.model else self.model
            idx = f"{self.lang}_{self.run_id}_{model_base}_turn-{self.turn}-{current_idx}"
            
            # Build message based on role
            message = self._build_message(data_input)
            
            # Build API request
            request = self._build_request(
                custom_id=idx,
                model=self.model,
                messages=message,
            )
            requests.append(request)
        
        # Save requests
        save_jsonl(requests, self.gen_file_path)
        logger.info("Generated %d requests saved to %s", len(requests), self.gen_file_path)
    
    def evaluate(self) -> None:
        """
        Evaluate generated dialogue responses and write evaluation results.
        
        This method checks if regeneration data exists and loads it if available;
        otherwise, loads the initially generated data. It then processes each data
        input by building evaluation requests and writes the results to the
        evaluation file.
        """
        # Determine which data to evaluate
        regen_path = Path(self.config.paths.completed_batches) / self.regen_file_path.replace(
            self.config.paths.batches_to_process + "/", ""
        )
        gen_path = Path(self.config.paths.completed_batches) / self.gen_file_path.replace(
            self.config.paths.batches_to_process + "/", ""
        )
        
        if regen_path.exists():
            logger.info("Regeneration data found. Evaluating regenerated responses.")
            data_to_evaluate = load_jsonl(str(regen_path))
        else:
            data_to_evaluate = load_jsonl(str(gen_path))
        
        # Build evaluation requests
        eval_requests = []
        for data_input in data_to_evaluate:
            eval_request = self._build_eval_request(data_input)
            eval_requests.append(eval_request)
        
        # Save evaluation requests
        save_jsonl(eval_requests, self.eval_file_path)
        logger.info("Evaluation requests saved to %s", self.eval_file_path)
    
    def regenerate(self) -> int:
        """
        Regenerate dialogue responses based on evaluation results.
        
        This method processes evaluation data to identify responses that need regeneration
        (marked by "No" in the response content). It builds new requests for these items
        and writes them to the regeneration file.
        
        Returns:
            Number of responses flagged for regeneration
        """
        regens_needed = 0
        edits = 0
        
        # Load data
        gen_path = Path(self.config.paths.completed_batches) / self.gen_file_path.replace(
            self.config.paths.batches_to_process + "/", ""
        )
        gen_data = load_jsonl(str(gen_path))
        gen_data_dict = {x['custom_id']: x for x in gen_data}
        
        # Handle prior regenerations
        regen_path = Path(self.config.paths.completed_batches) / self.regen_file_path.replace(
            self.config.paths.batches_to_process + "/", ""
        )
        timestamp = datetime.now().strftime("%H%M%S")
        
        if regen_path.exists():
            prior_regen_requests = load_jsonl(
                str(Path(self.config.paths.batches_to_process) / self.regen_file_path.replace(
                    self.config.paths.batches_to_process + "/", ""
                ))
            )
            prior_regen_data = load_jsonl(str(regen_path))
            # Archive old regeneration file
            new_regen_filename = str(regen_path).replace('.jsonl', f'_{timestamp}.jsonl')
            os.rename(str(regen_path), new_regen_filename)
        else:
            prior_regen_data = None
            prior_regen_requests = None
        
        # Load evaluation data
        eval_path = Path(self.config.paths.completed_batches) / self.eval_file_path.replace(
            self.config.paths.batches_to_process + "/", ""
        )
        eval_data = load_jsonl(str(eval_path))
        
        # Archive evaluation file
        new_eval_filename = str(eval_path).replace('.jsonl', f'_{timestamp}.jsonl')
        os.rename(str(eval_path), new_eval_filename)
        
        logger.info("Evaluating %d responses for regeneration. Saved logs with timestamp %s.", len(eval_data), timestamp)
        
        # Process evaluations and build regeneration requests
        regen_requests = []
        
        for current_idx, eval_input in enumerate(eval_data):
            true_idx = int(eval_input["custom_id"].split("-")[-1])
            dialogue_data = self.data[true_idx]
            
            eval_response = eval_input["response"]["body"]["choices"][0]["message"]["content"]
            
            if "Yes" not in eval_response:
                regens_needed += 1
                
                # Get prior response and message
                if prior_regen_data:
                    regen_res = prior_regen_data[current_idx]['response']['body']['choices'][0]['message']['content']
                    regen_request_message = prior_regen_requests[current_idx]["body"]["messages"]
                else:
                    regen_res = gen_data_dict[eval_input["custom_id"]]['response']['body']['choices'][0]['message']['content']
                    regen_request_message = self._build_message(dialogue_data)
                
                if not dialogue_data.get('ended', False):
                    idx = eval_input["custom_id"]
                    message = regen_request_message.copy()
                    
                    # Append feedback to message
                    message[-1]['content'] += (
                        f"\nPrior failed generation attempt was:\n"
                        f"{self.role}:{regen_res}\n"
                        f"Feedback from this previous generation:{eval_response}\n"
                    )
                    
                    request = self._build_request(
                        custom_id=idx,
                        model=self.model,
                        messages=message,
                    )
                    regen_requests.append(request)
                    
                    # Mark as ended to avoid continuation
                    gen_data_dict[eval_input["custom_id"]]['response']['body']['choices'][0]['message']['content'] = "END_OF_DIALOGUE"
                    edits += 1
            
            elif prior_regen_data:
                # Use prior regeneration if evaluation passed
                if eval_input["custom_id"] not in gen_data_dict:
                    raise ValueError(f"Data input {eval_input['custom_id']} not found in generated data.")
                gen_data_dict[eval_input["custom_id"]] = prior_regen_data[current_idx]
                edits += 1
        
        # Save regeneration requests
        if regen_requests:
            save_jsonl(regen_requests, self.regen_file_path)
        
        # Update original file if edits were made
        if edits > 0:
            save_jsonl(list(gen_data_dict.values()), str(gen_path))
        
        return regens_needed
    
    def _build_message(self, data_input: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build message list for API request based on role and dialogue context.
        
        Args:
            data_input: Dialogue data input
            
        Returns:
            List of message dictionaries
        """
        message = [{'role': 'system', 'content': self.sys_prompt}]
        
        if self.role == 'user':
            # Build user message with scene and dialogue context
            scene = data_input.get('scene', '')
            dialogue = data_input.get('dialogue', [])
            
            dialogue_text = '\n'.join([
                f"{turn['role']}: {turn['content'].replace(chr(10), '')}"
                for turn in dialogue
            ])
            
            user_content = (
                f"The scene is as follows: {scene}\n"
                f"The Dialogue is as follows:\n{dialogue_text}\n\n"
                f"The next user response is?"
            )
            
            message.append({
                'role': 'user',
                'content': user_content
            })
        else:
            # For assistant, use dialogue directly
            message.extend(data_input.get('dialogue', []))
        
        return message
    
    def _build_request(
        self,
        custom_id: str,
        model: str,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Build an API request dictionary.
        
        Args:
            custom_id: Custom identifier for the request
            model: Model identifier
            messages: List of message dictionaries
            
        Returns:
            Request dictionary
        """
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
                "frequency_penalty": self.model_config.frequency_penalty,
                "presence_penalty": self.model_config.presence_penalty,
                "max_tokens": self.model_config.max_tokens,
            }
        }
    
    def _build_eval_request(self, data_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build an evaluation request for a generated response.
        
        Args:
            data_input: Data containing the response to evaluate
            
        Returns:
            Evaluation request dictionary
        """
        # Extract index to find original dialogue context
        true_idx = int(data_input["custom_id"].split("-")[-1])
        dialogue_data = self.data[true_idx]
        
        # Build dialogue context string
        scene = dialogue_data.get('scene', '')
        dialogue = dialogue_data.get('dialogue', [])
        
        dialogue_context = f"The scene is as follows: {scene}\n\nThe Dialogue is as follows:\n"
        for turn in dialogue:
            dialogue_context += f"{turn['role']}: {turn['content']}\n"
        
        # Add generated response
        generated_response = data_input["response"]["body"]["choices"][0]["message"]["content"]
        dialogue_context += f"The response to evaluate is:\n{self.role}: {generated_response}"
        
        # Build evaluation message
        eval_messages = [
            {"role": "system", "content": EVALUATE_PROMPT},
            {"role": "user", "content": dialogue_context}
        ]
        
        # Use Gemini for evaluation with lower temperature
        eval_model_config = ModelConfig(temperature=0.1, max_tokens=512)
        
        return {
            "custom_id": data_input["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gemini-2.0-flash",
                "messages": eval_messages,
                "temperature": eval_model_config.temperature,
                "max_tokens": eval_model_config.max_tokens,
            }
        }
