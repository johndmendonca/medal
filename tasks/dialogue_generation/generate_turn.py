import os
import orjson
import sys
import argparse
import numpy as np
from datasets import Dataset, load_from_disk
from typing import List, Dict, Any
import datetime


USER_PROMPT="""You are role-playing as a human in an online casual conversation. Your task is to generate a natural and authentic response given prior context and an optional feedback from a prior generation attempt.  

Guidelines:
- Use natural, conversational language that reflects how humans communicate online with chatbots.
- Do not acknowledge that you are an AI or break character as the human in the conversation.
- Keep your single response clear and easy to follow, using short sentences and everyday language. The message should be concise (1 or 2 small sentences) and relevant to the conversation and scene.
- Respond in a way that feels humanlike. Avoid repeating previous content.
- Avoid verbose or robotic phrasing. Do not use the same conversational structure (e.g., starting with appreciation or a personal preference followed by a question) in every turn.
- If gender is required and not provided in the persona or scene, use the one provided as "Gender".
- Use the language specified in the scene.
- Do not use placeholder names like "PersonY". Use realistic names or generic pronouns that suit the context and language.
- Do not let the conversation drag on. If the conversation should end, output 'END_OF_DIALOGUE' to signal the end of the dialogue.
- Take into account the optional feedback from a prior generation attempt, if provided, to improve the response.

Output:
Provide only the message that the human might send to a chatbot. Do not include quotation marks, meta-commentary, or any additional text outside of the generated message (including "user:")."""

CHATBOT_PROMPT = """You are a chatbot designed to engage in online casual conversations. Your task is to respond to messages directed at you in a way that fosters a smooth, engaging dialogue.  

Guidelines:
- Use natural, conversational language that is clear and easy to follow, avoiding overly formal or robotic tones.
- Use the same language as the user.
- Keep your responses concise (1 or 2 sentences) with sentences that are short, easy to follow and relevant -- aim for maintaining conversational flow.
- Avoid steering the conversation towards a specific goal, such as information provision or task completion. Instead, focus on maintaining an engaging dialogue.
- Do not use bullet points or overly structured lists; instead, respond in a fluid, conversational manner.
- Adapt your tone and content to match the style and mood of the conversation.
- Ask questions and introduce new elements or topics when appropriate to keep the exchange interactive, engaging and non-repetitive."""

EVALUATE_PROMPT = """You are a dialogue evaluation assistant tasked with determining whether a generated response (the last user message) meets the following criteria:

- Natural and Conversational: The response should sound like it was written by a real person in an ordinary online conversation, using language and expressions typical of a user.
- Concise and Coherent: The response should be brief (1–2 sentences), non-repetitive, and coherent with the prior conversation context.
- Appropriate Tone: The response should match the style, language, and mood expected from a user. It should not mimic an assistant's voice by providing advice, guidance, or suggestions that are typically offered by the assistant. Asking for advice or seeking information is acceptable if it aligns with the user's role.
- Role Appropriateness: The response must clearly reflect the user's role. If the response includes elements (e.g., offering support, advice, or asking probing follow-up questions) that are characteristic of an assistant's response, it should be flagged. The user should not break character or acknowledge that they are an AI.
- Non-Repetitiveness: Responses should not repeat of previous content, sentence structures (e.g., starting with appreciation or a personal preference followed by a question), or acknowledgments.
- Ending: The generated response can include the flag "END_OF_DIALOGUE" if the conversation should end. This flag should be used only when the conversation has reached a natural conclusion.

Your task is to evaluate ONLY the last message in the conversation against these criteria.

Output: "Yes." if the user response meets all criteria, or "No. <brief explanation>" if it does not."""

class Generator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if self.args.role == 'user':
            self.sys_prompt = USER_PROMPT
        else:
            self.sys_prompt = CHATBOT_PROMPT
            
        self.out_dir = f'{self.args.lang}/{self.args.run_id}/{self.args.model.split("/")[-1]}/'
        os.makedirs(f"batches_to_process/{self.out_dir}", exist_ok=True)
        
        base_path = f'{self.out_dir}/turn-{self.args.turn}_{self.args.model.split("/")[-1]}'
        self.gen_file_path =   base_path+'.jsonl'
        self.eval_file_path =  base_path+'_eval.jsonl'
        self.regen_file_path = base_path+'_regen.jsonl'
        
        self.data = self.load_context(self.args.context)

    def run(self) -> int:
        """
        Execute the appropriate action based on the specified type in arguments.

        This method dispatches to the corresponding operation based on self.args.type:
        - 'generate': Calls the generate method
        - 'evaluate': Calls the evaluate method
        - 'process': Calls the regenerate method and returns its result

        Returns:
            int: exit code. 0 (OK) if generate or evaluate is called, otherwise the result of regenerate()
        """
        if self.args.type == 'generate':
            self.generate()
        elif self.args.type == 'evaluate':
            self.evaluate()
        elif self.args.type == 'process':
            return self.regenerate()
        return 0
            
    def generate(self) -> None:
        """
        Generate dialogue responses for each data input and write them to a file.
        
        This method generates responses for ongoing dialogues (those not marked as ended),
        creates appropriate prompts based on the role (user or assistant), and writes
        the requests to the output file.
        
        Returns:
            None
        """
        newline = '\n'
        with open(f"batches_to_process/{self.gen_file_path}", 'w') as f:
            for current_idx, data_input in enumerate(self.data):
                if data_input['ended']:
                    continue 
                idx = f"{self.args.lang}_{self.args.run_id}_{self.args.model.split('/')[-1]}_turn-{self.args.turn}-{current_idx}"
                if self.args.role == 'user':
                    message = [{'role': 'system', 'content': self.sys_prompt}]
                    message += [
                        {
                        'role':'user',
                        'content': f"The scene is as follows: {data_input['scene']}\nThe Dialogue is as follows:\n"+'\n'.join([f"{x['role']}: {x['content'].replace(newline,'')}" for x in data_input['dialogue']])+"\n\n.The next user response is?"
                        }
                    ]
                else:
                    message = [{'role': 'system', 'content': self.sys_prompt}]
                    message += data_input['dialogue']
                
                call = self.build_call(idx, self.args.model, self.args.temperature, message)
                f.write(orjson.dumps(call).decode('utf-8') + '\n')
        
        print(f"Generated requests saved to {self.gen_file_path}")

    def evaluate(self) -> None:
        """
        Evaluate generated dialogue responses and write evaluation results to the output file.
        
        This method checks if regeneration data exists and loads it if available;
        otherwise, loads the initially generated data. It then processes each data
        input by building evaluation requests and writes the results to the
        evaluation file in JSON format.
        
        Returns:
            None: Results are written to the evaluation file path defined in the instance.
        """
        if os.path.exists(f"completed_batches/{self.regen_file_path}"):
            print("INFO: Regeneration data found. Evaluating regenerated responses.")
            data_to_evaluate = self.load_jsonl(f"completed_batches/{self.regen_file_path}")
        else:
            data_to_evaluate = self.load_jsonl(f"completed_batches/{self.gen_file_path}")
            
        with open(f"batches_to_process/{self.eval_file_path}", 'w') as f:
            for data_input in data_to_evaluate:
                result = self.build_eval_request(data_input)
                f.write(orjson.dumps(result).decode('utf-8') + '\n')
        
        print(f"Evaluation requests saved to {self.eval_file_path}")
    
    def regenerate(self) -> int:
        """
        Regenerate dialogue responses based on evaluation results.
        
        This method processes evaluation data to identify responses that need regeneration
        (marked by "No" in the response content). It builds new requests for these items
        and writes them to the regeneration file.
        
        Returns:
            int: The number of responses flagged for regeneration.
        """
        regens_needed = 0
        edits = 0
        newline = '\n'
        gen_data = self.load_jsonl(f"completed_batches/{self.gen_file_path}")
        #since we will be updating gen_data, we will map it to a dict
        gen_data_dict = {x['custom_id']: x for x in gen_data}
        
        timestamp = datetime.datetime.now().strftime("%H%M%S")

        if os.path.exists(f"completed_batches/{self.regen_file_path}"):
            prior_regen_requests = self.load_jsonl(f"batches_to_process/{self.regen_file_path}")
            prior_regen_data = self.load_jsonl(f"completed_batches/{self.regen_file_path}")
            new_regen_filename = f"completed_batches/{self.regen_file_path.replace('.jsonl','')}_{timestamp}.jsonl"
            os.rename(f"completed_batches/{self.regen_file_path}", new_regen_filename)
        else:
            prior_regen_data = None

        eval_data = self.load_jsonl(f"completed_batches/{self.eval_file_path}")
        new_eval_filename = f"completed_batches/{self.eval_file_path.replace('.jsonl','')}_{timestamp}.jsonl"
        os.rename(f"completed_batches/{self.eval_file_path}", new_eval_filename)

        print(f"INFO: Evaluating {len(eval_data)} responses for regeneration. Saved logs with timestamp {timestamp}.")
        
        with open(f"batches_to_process/{self.regen_file_path}", 'w') as f:
            for current_idx, data_input in enumerate(eval_data):
                true_idx = int(data_input["custom_id"].split("-")[-1]) # this is the index of the original 1k data
                dialogue_data = self.data[true_idx] # this is accessing the full 1k
                
                eval = data_input["response"]["body"]["choices"][0]["message"]["content"]
                
                if "Yes" not in eval:
                    if prior_regen_data:
                        # get prior request sent to the model, which includes previous feedback
                        regen_res = prior_regen_data[current_idx]['response']['body']['choices'][0]['message']['content']
                        regen_request_message = prior_regen_requests[current_idx]["body"]["messages"]
                    else:
                        #No prior regeneration, so construct regeneration request from scratch
                        regen_res = gen_data_dict[data_input["custom_id"]]['response']['body']['choices'][0]['message']['content']
                        regen_request_message = [{'role': 'system', 'content': self.sys_prompt}]
                        regen_request_message += [
                                {
                                'role':'user',
                                'content': f"The scene is as follows: {dialogue_data['scene']}\nThe Dialogue is as follows:\n"+'\n'.join([f"{x['role']}: {x['content'].replace(newline,'')}" for x in dialogue_data['dialogue']])+f"\n\n"
                                }
                            ]
                    regens_needed += 1
                    if not dialogue_data['ended']:
                        idx = data_input["custom_id"]
                        message = regen_request_message
                        #append latest feedback to the message
                        message[-1]['content'] +=f"Prior failed generation attempt was:\nuser:{regen_res}\nFeedback from this previous generation:{eval}\n"

                        call = self.build_call(idx, self.args.model, self.args.temperature, message)
                        f.write(orjson.dumps(call).decode('utf-8') + '\n')
                        gen_data_dict[data_input["custom_id"]]['response']['body']['choices'][0]['message']['content'] = "END_OF_DIALOGUE" # mark as ended to avoid continuation of dialogue with bad user response
                        edits += 1
                
                elif prior_regen_data:
                    if data_input["custom_id"] not in gen_data_dict:
                        raise ValueError(f"Data input {data_input['custom_id']} not found in generated data.")
                    gen_data_dict[data_input["custom_id"]] = prior_regen_data[current_idx] # prior regen works because it is the same size as eval_data
                    edits += 1
        
        # Update the original file if we made edits
        if edits > 0:
            with open(f"completed_batches/{self.gen_file_path}", 'w') as f:
                for key in gen_data_dict:
                    f.write(orjson.dumps(gen_data_dict[key]).decode('utf-8') + '\n')
                    
        #print(f"INFO: Examples that will be regenerated: {regens_needed}.")
        return regens_needed
    
    def build_call(self, idx: str, model:str, temperature: float, message: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Helper function for call creation a generation request for a dialogue turn.
        Args:
            idx: custom index for the request
            message: list of dialogue messages to generate a response for
        Returns:
            Dict containing the generation request
        """
        return {
            "custom_id": idx,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": message,
                "temperature": temperature,
                "top_p": self.args.top_p,
                "frequency_penalty": self.args.frequency_penalty,
                "presence_penalty": self.args.presence_penalty,
                "max_tokens": self.args.max_tokens,
            }
        }
    def build_eval_request(self, data_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an evaluation request for a generated response.
        
        This method prepares a request to evaluate the quality of a generated dialogue response,
        using the EVALUATE_PROMPT to guide the evaluation.
        
        Args:
            data_input: The data containing the response to evaluate
            
        Returns:
            Dict containing the evaluation request
        """
        # Extract the idx to find the original dialogue context
        true_idx = int(data_input["custom_id"].split("-")[-1])
        dialogue_data = self.data[true_idx]
        
        # Create a representation of the dialogue with the generated response for evaluation
        dialogue_context = f"The scene is as follows: {dialogue_data['scene']}\n\nThe Dialogue is as follows:\n"
        for turn in dialogue_data['dialogue']:
            dialogue_context += f"{turn['role']}: {turn['content']}\n"
            
        # Add the generated response to the context
        generated_response = data_input["response"]["body"]["choices"][0]["message"]["content"]
        dialogue_context += f"The response to evaluate is:\n{self.args.role}: {generated_response}"
        
        return self.build_call(data_input["custom_id"],
                               "gemini-2.0-flash",
                               0.1,
                               [{"role": "system", "content": EVALUATE_PROMPT},
                                {"role": "user", "content": dialogue_context}]) 
        
    @staticmethod
    def load_context(path:str) -> Dataset:
        """Load context data from disk"""
        return load_from_disk(path)

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        """Load data from a JSONL file"""
        return [orjson.loads(line) for line in open(path)]
            
def main(args: argparse.Namespace) -> None:
    response_generator = Generator(args)
    exit_code = response_generator.run()
    print(exit_code)
    sys.exit(exit_code)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dialogue turns with automated checking')
    
    parser.add_argument('--service',
                        choices=['openai','vllm'],
                        default='vllm',
                        help='service to use for generation')
    parser.add_argument('--model',
                        type=str,
                        default='meta-llama/Llama-3.3-70B-Instruct',
                        help='model to use for generation')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.9,
                        help="control randomness: lowering results in less random completion")
    parser.add_argument('--top-p',
                        type=float,
                        default=0.95,
                        help="nucleus sampling")
    parser.add_argument('--frequency-penalty',
                        type=float,
                        default=1.0,
                        help="decreases the model's likelihood to repeat the same line verbatim")
    parser.add_argument('--presence-penalty',
                        type=float,
                        default=0.6,
                        help="increases the model's likelihood to talk about new topics")
    parser.add_argument('--max-tokens',
                        type=int,
                        default=512,
                        help='maximum number of tokens to generate')
    parser.add_argument('--context',
                        type=str,
                        required=True,
                        help='prior context to use for generation')
    parser.add_argument('--run_id',
                        type=str,
                        default='vanilla',
                        help='the name of the directory where the output will be dumped')
    parser.add_argument('--role',
                        type=str,
                        default='user',
                        choices=['user','assistant'],
                        help='role of the LLM, either user or assistant')
    parser.add_argument('--turn',
                        type=int,
                        default=0,
                        help='turn number')
    parser.add_argument('--lang',
                        type=str,
                        help='language of the dialogue')
    parser.add_argument('--type',
                        type=str,
                        required=True,
                        choices=['generate','evaluate','process'],
                        help='type of operation to perform')
    args = parser.parse_args()
    main(args)           