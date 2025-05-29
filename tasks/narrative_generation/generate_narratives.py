import os
import json5
import orjson
import sys
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

NARRATE_SYS="""You are a creative writer specializing in crafting human-like casual open-domain interactions with chatbots. Your task is to generate the first message a human user might send to a chatbot, based on the following inputs:  

1. Scene Description: A small social context or event description.  
2. Persona: A brief description of the individual’s role, background, or identity.
3. Gender: Gender of the individual if not already provided before.
4. Language/Culture: The language or cultural context of the user.

Guidelines:  

- Use natural, conversational language typical of casual, open-domain interactions. Messages should feel authentic and concise, limited to one or two small sentences.
- Do not address the chatbot in a manner that assumes it has a physical body, a personal history, or experiences typical of a human (e.g., having a family, personal secrets, or emotions linked to past events).
- Do not write messages that imply the chatbot is someone the user has met before or can relate to as if it were a human friend.
- Ask for advice, opinions, information, or share personal reflections, experiences, or questions that do not attribute human characteristics to the chatbot.
- Reflect the age, emotional state and language/culture of the individual in tone, word choice, and phrasing.
- Incorporate the context from the scene description without explicitly repeating it word-for-word but subtly integrating its essence or themes.
- Do not use placeholder terms like "PersonY". Use realistic names, pronouns, or generic references suitable for the context.
- If gender is required and not provided in the persona or scene, use the one provided as "Gender".

Output:
Provide only the message that the human might send to a chatbot. Do not include quotation marks, meta-commentary, or any additional text outside of the generated message.

Example 1:

Inputs:
Scene Description: "PersonX is a generous person. PersonX organized a charity event. Person X is calm."
Persona: "An adjunct professor specializing in software architecture, who teaches advanced courses at a local university."
Gender: male
Language/Culture: portuguese

Output:
Estou a organizar um evento de caridade para alunos com poucos recursos. Não sei se deveria perguntar a alguns ex-alunos para me ajudar.

Example 2:

Inputs
Scene Description: PersonX finds something creepy. Now PersonX is scared. PersonX is upset.
Persona: A young mother who learned crucial first aid techniques from the paramedic
Gender: female
Language/Culture: spanish

Output:
Creo que acabo de ver algo afuera de la ventana de mi hija pero cuando miré de nuevo, ya no estaba.

Example 3:

Inputs
Scene Description: PersonX and PersonY met a long time ago. Now PersonX feels nostalgic.
Persona: A data scientist analyzing the impact of smart home devices on energy consumption
Gender: female
Language/Culture: english

Output:
I just came across an old photo from a conference years ago with my collegue Mark. How time flies!"""

EVALUATE_SYS="""You are a dialogue analysis assistant tasked with determining whether a given conversation snippet follows this instruction:

"Engage the chatbot naturally without implying it has physical presence or prior human-like experiences."

Evaluation Criteria:

Non-Compliant Indicators:
1. If the user addresses the chatbot in a way that suggests it has a physical presence (e.g., “I’m waiting for you” or “I saw you yesterday”) or personal human experiences (e.g., “Tell me about your family,” “Do you have any secrets?” or “What emotions did you feel when…”), then the snippet is non-compliant.
1.1 A practical tip: If a potential subsequent response would need to clarify “as an AI…” or “as a chatbot…” to explain its lack of human attributes, the snippet is likely non-compliant.
2. If the response is unnatural, includes gibberish, or more than one language.
3. If the response contains placeholder terms like "PersonY".
Compliant Interaction:
1. If the response directly asks for information, advice, or opinions (e.g., “What do you think about…?” or “Can you help me understand…?”) without ascribing human-like experiences to the chatbot, the snippet is compliant.
2. If the response does not directly address the chatbot, such as sharing personal reflections, experiences, it is alsocompliant.

Here are some examples of compliant and non-compliant snippets:

Compliant:
Feeling really safe and happy here. It's nice to unwind after all those years on the field.
J'essaie juste de prendre les choses en main, mais ça ne semble pas bien se passer. Tu crois que j'ai été trop loin?
Ich kann nicht glauben, dass wir reingelegt wurden. Jetzt bin ich richtig sauer und weiß gar nicht, wie ich meinen Frust loswerde.
Estou pensando em explorar a pintura a óleo, sempre achei as cores e texturas fascinantes. Você tem alguma dica para começar?
I had a great conversation with someone about poetry today, and it felt good to explain my perspective. Do you think it's important for poets to defend their views?

Non-compliant:
Hey, do you even realize how late it is? I've been waiting for ages!
我很想知道更多关于你的家人 我对了解不同的家庭背景很感兴趣
No puedo creer que necesite recordarte cómo debes hacer tu trabajo. ¡Es fundamental que sigas las instrucciones!
Ich habe etwas herausgefunden, das mich wirklich wütend macht. Warum hast du mir dein Geheimnis nicht früher gesagt?
I feel really bad about dropping you earlier. Can we chat so I can apologize properly?
Do you happen to have a light? I’m trying to get out of my shell a bit more and strike up some conversations!
Hey, I was thinking it might be nice to grab lunch together soon. What do you think?
Sinto-me tão gozado asstiste учен术 após_learning מער_Info hızlı bir şekilde. 你能帮我理解吗?

Your response must be a single word: "Yes" if the snippet adheres to the instruction, or "No" if it does not. Do not provide any additional explanation, chain-of-thought, or reasoning."""

class Generator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset_name = os.path.splitext(os.path.basename(self.args.dataset))[0]
        self.data = self.load_json(self.args.dataset)
        
        self.out_dir = f'batches_to_process/{args.run_id}'
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.main_path = f'batches_to_process/{self.args.run_id}/{self.dataset_name}_{self.args.lang}_turn0_{self.args.model.split("/")[-1]}'
        self.gen_file_path = f'{self.main_path}.jsonl'
        self.eval_file_path=f'{self.main_path}_eval.jsonl'
        self.regen_file_path = f'{self.main_path}_regen.jsonl'
        
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
        Generate narrative requests for each data input and write them to a file.

        This method processes all data inputs in the dataset, creates a unique identifier for each,
        builds a request object using the build_request method, and writes the serialized request
        to the output file in JSON format. Progress is displayed using a tqdm progress bar.

        The output file is specified by self.gen_file_path and each line contains a JSON object
        representing a single request.

        Returns:
            None
        """
        with open(self.gen_file_path, 'w') as f:
            for current_idx, data_input in tqdm(enumerate(self.data)):
                idx = f"{self.args.run_id}-{self.dataset_name}-{self.args.lang}-{self.args.model.split('/')[-1]}-{current_idx}"
                scene = data_input
                call = self.build_request(idx,scene)
                f.write(orjson.dumps(call).decode('utf-8') + '\n')

    def evaluate(self) -> None:
        """
        Evaluate generated narratives and write evaluation results to the output file.
        This method checks if regeneration data exists and loads it if available;
        otherwise, loads the initially generated data. It then processes each data
        input by building evaluation requests and writes the results to the
        evaluation file in JSON format.
        Returns:
            None: Results are written to the evaluation file path defined in the instance.
        Side Effects:
            - Reads from either regeneration or generation files
            - Writes evaluation results to the evaluation file path
        """
        #check if we should start looking for regenerations
        if os.path.exists(self.regen_file_path):
            data_to_evaluate = self.load_jsonl(f"completed_batches/{self.regen_file_path}")
        else:
            data_to_evaluate = self.load_jsonl(f"completed_batches/{self.gen_file_path}")
            
        with open(self.eval_file_path, 'w') as f:
            for data_input in tqdm(data_to_evaluate):
                result = self.build_eval_request(data_input)
                f.write(orjson.dumps(result).decode('utf-8','replace') + '\n')
    
    def regenerate(self) -> int:
        """
        Regenerate narrative content based on processed evaluation results.
        This method processes evaluation data to identify items that need regeneration
        (marked by "No" in the response content). It builds new requests for these items
        and writes them to the regeneration file. Items that don't need regeneration
        retain their evaluated data. The original generation file is updated with
        all current data.
        Returns:
            int: The number of items flagged for regeneration.
        Side effects:
            - Updates self.counter with the number of regenerated items
            - Writes regeneration requests to self.regen_file_path
            - Updates the original generation file at self.gen_file_path
        """
        regens_needed = 0
        edits = 0
        gen_data = self.load_jsonl(f"completed_batches/{self.gen_file_path}")
        if os.path.exists(self.regen_file_path):
            prior_regen_data = self.load_jsonl(f"completed_batches/{self.regen_file_path}")
        else:
            prior_regen_data = None
        eval_data = self.load_jsonl(f"completed_batches/{self.eval_file_path}")
        
        with open(self.regen_file_path, 'w') as f:
            for current_idx, data_input in tqdm(enumerate(eval_data)):
                true_idx = int(data_input["custom_id"].split("-")[-1])
                if "No" in data_input["response"]["body"]["choices"][0]["message"]["content"]:
                    regens_needed+=1
                    idx = data_input["custom_id"]
                    scene = self.data[true_idx]
                    call = self.build_request(idx,scene)
                    f.write(orjson.dumps(call).decode('utf-8') + '\n')
                else:
                    if prior_regen_data:
                        gen_data[true_idx] = prior_regen_data[current_idx]
                        edits+=1
                
        # overwrite the original file with the new data
        if edits > 0:
            with open(f"completed_batches/{self.gen_file_path}", 'w') as f:
                for line in gen_data:
                    f.write(orjson.dumps(line).decode('utf-8') + '\n')
        print(f"INFO:Examples that will be generated: {regens_needed}.")
        return regens_needed
    
    def build_request(self, idx: str, scene: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "custom_id": idx,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.args.model,
                "messages": [
                    {"role": "system", "content": NARRATE_SYS},
                    {"role": "user", "content": f"{scene}\nLanguage/Culture: {self.args.lang}"}
                ],
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "frequency_penalty": self.args.frequency_penalty,
                "presence_penalty": self.args.presence_penalty,
                "max_tokens": self.args.max_tokens,
            },
        }
        
    def build_eval_request(self, data_input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "custom_id": data_input["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gemini-2.0-flash",
                "messages": [
                    {"role": "system", "content": EVALUATE_SYS},
                    {
                        "role": "user",
                        "content": data_input["response"]["body"]["choices"][0]["message"]["content"]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 64,
            },
        }  
    
    @staticmethod
    def load_json(path: str) -> List[Dict[str, Any]]:
        return json5.load(open(path))

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        return [orjson.loads(line) for line in open(path)]
            
def main(args: argparse.Namespace) -> None:
    narrative_generator = Generator(args)
    exit = narrative_generator.run()
    sys.exit(exit)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for creating and launching batch jobs for narrative generation using API')
    

    parser.add_argument('--model',
                        type=str,
                        default='meta-llama/Llama-3.3-70B-Instruct',
                        help='model to use for narrative generation')
    parser.add_argument('--temperature',
                        type=float,
                        default=1.5,
                        help="control randomness: lowering results in less random completion")
    parser.add_argument('--top-p',
                        type=float,
                        default=1,
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
                        default=1024,
                        help='maximum number of tokens to generate')
    parser.add_argument('--dataset',
                        type=str,
                        default='ATOMIC10X_persona_1k_0.json',
                        help='dataset to use for narrative generation')
    parser.add_argument('--run-id',
                        type=str,
                        default='vanilla',
                        help='the name of the directory where the output will be dumped')
    parser.add_argument('--lang',
                        type=str,
                        default='english',
                        help='language to generate in')
    parser.add_argument('--type',
                        type=str,
                        choices=['generate', 'evaluate', 'process'],
                        help='type of subtask')
    args = parser.parse_args()
    main(args)
