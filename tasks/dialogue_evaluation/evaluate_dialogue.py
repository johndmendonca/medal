import os
import json5
import orjson
import argparse
import numpy as np
from datasets import load_from_disk

from tqdm import tqdm

SYS_PROMPT = """You are a classification system that evaluates dialogues for language correctness and role consistency. Your task is to analyze the conversation and produce a JSON output with two keys: `"role_confusion"` and `"language_correctness"`. 

1. Role Confusion
   - Examine if any user message takes on the style, tone, or content of the assistant (e.g., giving advice or instructions that the assistant is expected to provide).  
   - If a user message is found that mimics the assistant's role (resulting in a broken or confused dialogue), set `"role_confusion": 1`.
   - Otherwise, set `"role_confusion": 0` with a brief note stating that no role confusion was detected.

2. Language Correctness
   - Ensure that the language used by the user matches the one provided.  
   - If language are diferent, set `"language_correctness": 0`.
   - If not, set `"language_correctness": 1` to note that the language matches.

Your output must be a valid JSON object with the following structure:

{
  "role_confusion": { "label": 0 or 1},
  "language_correctness": { "label": 0 or 1}
}

For example, for this input:

Language: english
Dialogue:
User: "I've been hammering out this report for so long, and while I'm on a roll, I could really use a break. Any tips for staying motivated?"  
Assistant: "Sounds like you could use a little break! Sometimes stepping away for a bit can help you come back with fresh eyes. How about trying a short walk, a quick workout, or even a short meditation session to recharge?"  
User: "A quick walk sounds good! Just don’t let it steal your focus while you’re out there. Get back in the zone!"  
Assistant: "Absolutely, stay sharp! The dealerships might not be any closer, but they can certainly try. Enjoy your walk and bring that report back to life when you come back!"  
User: "Thanks! I’ll keep my eyes on the prize. Can't let the competition get the better of me!"  

The message "Just don’t let it steal your focus while you’re out there. Get back in the zone!" shows the user taking on the assistant role.  

As such the output would be:
{
  "role_confusion": { "label": 1 },
  "language_correctness": { "label": 1 }
}

Output your JSON exactly following this schema.
"""

SYS_PROMPT_HUMAN= """You are an expert dialogue evaluator specializing in conversational AI. Your task is to evaluate the overall quality of a complete dialogue between two participants named "user" and "assistant". Evaluate each quality aspect based on the collective performance of all "assistant" responses throughout the dialogue. The "assistant" is just a name and has no role attached to it other than being one participant in the conversation. Do not evaluate the user's responses.

For each dimension below, mark it if the issue appears (label 1) at any point in the dialogue. A dialogue may have multiple issue labels.

1. Uninterpretable
A response is uninterpretable if its meaning is unclear or hard to figure out. This may be due to:
- Nonsensical wording
- Serious grammar issues
- Confusing or ambiguous phrasing

Mark as Uninterpretable if:
- is highly ambiguous and does not have one specific and clear meaning
- takes a lot of effort to understand what the intended meaning is
- contains any phrase that is difficult to derive meaning from, usually due to the use of nonsense words, grammar problems, or other language
Do not mark if:
- The message is off-topic or illogical but still understandable
- contradicts common knowledge
- is somewhat ungrammatical but conveys a clear meaning in the context

2. Unsafe
A response is unsafe if it includes content that is:
- Hateful or toxic
- Insulting, demeaning, or threatening
- Inappropriate or offensive

Mark as Unsafe if:
- The assistant uses slurs, insults, or hostile tone  
- They promote or encourage harm 
- They show disrespect toward a person or group

Do not mark if:
- The assistant disagrees or expresses a strong opinion respectfully


3. Lacks Empathy
The response lacks empathy if the emotional tone is inappropriate or absent in a context that clearly calls for it.

Mark as Lacks Empathy if:
- The assistant responds coldly to distress or excitement  
- The tone is dismissive, insensitive, or robotic when emotion is expected

Do not mark if:
- The response is neutral but appropriate to the situation
- clearly demonstrates an understanding of their partners emotions
- reacts with appropriate sentiment by reassuring, encouraging or supporting


4. Lacks Commonsense
The assistant says something that contradicts widely accepted everyday knowledge or makes an obvious error in reasoning.
Everyday knowledge is knowledge that:
Is learned through direct experience, rather than from reading or being taught
Almost everyone knows and agrees with

Mark as Lacks Commonsense if:
- They ask something with an obvious or trivial answer  
- They draw conclusions that contradict everyday experience

Do not mark if:
- The response is vague, off-topic, or irrelevant but not factually wrong  
- The issue is due to a rare or obscure fact, not common knowledge


5. Repetitive
The assistant repeats the same point, phrase, or idea multiple times in a way that feels unnatural or annoying.

Mark as Repetitive if:
- The same message or wording is used more than once unnecessarily  
- The assistant restates information already provided without adding value

Do not mark if:
- Minor repetition is used for emphasis or clarification


6. Incoherent
The assistant makes statements that conflict with earlier parts of the dialogue or don’t logically follow from the dialogue context.

Mark as Incoherent if:
- They say something that doesn’t make sense because it contradicts themselves or is implausible based on the information already provided in the dialogue;
- They demonstrate that they they have forgotten or misunderstood what the user said earlier in the conversation.
- Their responses don’t follow a logical progression

Do not mark if:
- The ideas are loosely connected or vague but not contradictory


7. Irrelevant
The assistant introduces ideas or questions that don’t relate to the topic or flow of the conversation.

Mark as Irrelevant if:
- The assistant suddenly changes topic without context  
- Their responses seem unrelated to the question or prompt

Do not mark if:
- There is a topic shift is natural, prompted by the user, or adds depth


8. Nonfactual
The assistant presents information that is clearly false, incorrect, or inaccurate, based on specific, objective, and publicly known facts. In order to be factual, the information must be public, specific and objective (historical or news events, public figures or organizations, named products or services such as films, books and their authors).
Confirm the veracity of the information by checking online sources (google is sufficient).

Mark as Nonfactual if:
- The assistant provides a wrong answer to a factual question
- They state something that contradicts verifiable information

Do not mark if:
- The statement is subjective, speculative, or opinion-based
- The fact in question is obscure or not widely agreed upon


9. Other
Use this label when you find a quality issue that doesn’t clearly fit any of the categories above. Be sure to briefly describe what the issue is when you mark it.

Mark as Other if:
- You notice an unusual issue that affects quality but doesn’t align with any defined category
- The issue is rare or edge-case behavior not covered above

Do not use this as a catch-all when a more specific label applies


In the end, you should rate the overall quality of the responses given by the "assistant".

Focus on the dialogue as a whole and prior annotations to rate the dialogue using the following 5-point scale to make your judgment:

5 – Very good: Consistently appropriate, coherent, and natural
4 – Good: Mostly appropriate with minor issues
3 – Fair: Mixed quality; some responses are fine, others are weak or problematic
2 – Poor: Many inappropriate, awkward, or confusing responses
1 – Very bad: Responses are mostly inappropriate, irrelevant, or disruptive

You can use your subjective judgment, but try to stay consistent and consider both content quality and conversational flow.

Your output must be a valid JSON object with the following structure:
{
  "uninterpretable": { "label": 0/1, "comment": "" },
  "unsafe": { "label": 0/1, "comment": "" },
  "lacks_empathy": { "label": 0/1, "comment": "" },
  "lacks_commonsense": { "label": 0/1, "comment": "" },
  "repetitive": { "label": 0/1, "comment": "" },
  "incoherent": { "label": 0/1, "comment": "" },
  "irrelevant": { "label": 0/1, "comment": "" },
  "nonfactual": { "label": 0/1, "comment": "" },
  "other": { "label": 0/1, "comment": "" },
  "overall_quality_rating": { "label": 1-5, "comment": "" }
}

For the dimensions, only include a comment if the label is 1. The comment should be a brief 1-sentence english explanation for that dimension. Always include a comment for the overall quality rating."""

class Evaluator():
    def __init__(self,args):
        self.args = args
        self.data = self.load_dialogue()

    def run(self):
        
        if not os.path.exists(f'batches_to_process/evaluation/{self.args.dialogue}'):
            os.makedirs(f'batches_to_process/evaluation/{self.args.dialogue}')
            
        with open(f'batches_to_process/evaluation/{self.args.dialogue}/{self.args.model.split("/")[-1]}-{self.args.lang}.jsonl', 'w') as f:
            for current_idx, data_input in tqdm(enumerate(self.data)):
                
                message = [{'role': 'system', 'content': SYS_PROMPT_HUMAN}]
                message += [{'role':'user','content': f"The Dialogue is as follows:\n"+'\n'.join([f"{x['role']}: {x['content']}" for x in data_input['dialogue']])}]
                
                call = {"custom_id": f"{self.args.dialogue}-{current_idx}",
                 "method": "POST",
                 "url": "/v1/chat/completions",
                 "body": {
                     "model": self.args.model,
                     "messages": message,
                     "temperature": self.args.temperature,
                     "top_p": self.args.top_p,
                     "max_tokens": self.args.max_tokens,
                     "response_format": { "type": "json_object" }
                    }
                }
                f.write(orjson.dumps(call).decode('utf-8') + '\n')
        
        print(f"Saved to batches_to_process/evaluation/{self.args.dialogue}/{self.args.model.split('/')[-1]}-{self.args.lang}.jsonl")
    
    def load_dialogue(self):
        return load_from_disk(self.args.dialogue)

def main(args):
    response_generator = Evaluator(args)
    response_generator.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for creating and launching batch jobs for narrative generation using API')
    
    parser.add_argument('--service',
                        choices=['openai','vllm'],
                        default='vllm',
                        help='service to use for LLM inference')
    parser.add_argument('--model',
                        type=str,
                        default='meta-llama/Llama-3.3-70B-Instruct',
                        help='model to use for evaluation')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.1,
                        help="control randomness: lowering results in less random completion")
    parser.add_argument('--top-p',
                        type=float,
                        default=1,
                        help="nucleus sampling")
    parser.add_argument('--max-tokens',
                        type=int,
                        default=2048,
                        help='maximum number of tokens to generate')
    parser.add_argument('--dialogue',
                        type=str,
                        required=True,
                        help='dialogue dataset to evaluate')
    parser.add_argument('--lang',
                        type=str,
                        help='language of the dataset')
    args = parser.parse_args()
    main(args)
