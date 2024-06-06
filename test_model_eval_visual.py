
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from safetensors.torch import load_file
from safetensors.torch import load_file as safe_load_file
import argparse
import lm_eval
from lm_eval.utils import make_table

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

parser = argparse.ArgumentParser(description='Model evaluator')
parser.add_argument("--model_name", type=str, default='/mnt/disk5/maktukmak/models/quant/vicuna-7b-v1.1')
args = parser.parse_args()


model_name = '/mnt/disk5/maktukmak/models/quant/vicuna-7b-v1.1'
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

model_name = 'lmsys/vicuna-7b-v1.1'
model2 = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompts = ["Somatic hypermutation allows the immune system to",
            "Hello, I'm a language model",
            "Translate from English to French: I'm very happy to see you",
            "Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change. Write a summary of the above text. Summary:",
            "Classify the text into neutral, negative or positive. Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen. Sentiment:",
            "Answer the question using the context below. Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors. Question: What modern tool is used to make gazpacho? Answer:",
            "There are 5 groups of students in the class. Each group has 4 students. How many students are there in the class?",
            ]

model = model.to('cuda')
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    gen = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)
    print(prompt)
    print(gen)


