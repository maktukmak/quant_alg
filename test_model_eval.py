
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from safetensors.torch import load_file
from safetensors.torch import load_file as safe_load_file
import argparse
import lm_eval
from lm_eval.utils import make_table


parser = argparse.ArgumentParser(description='Model evaluator')
parser.add_argument("--model_name", type=str, default='facebook/opt-125m')
parser.add_argument("--batch_size", type=int, default=16, help='batch_size')
args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code = True, torch_dtype=torch.float16).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

#model = torch.load("./models/quant/" + 'opt_125m').to('cuda')
# prompt = "Somatic hypermutation allows the immune system to"
# inputs = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
# outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer, dtype=torch.float16, max_length=tokenizer.model_max_length,
            batch_size=args.batch_size, trust_remote_code=True)

results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["mmlu"],
    task_manager=lm_eval.tasks.TaskManager(),)


if results is not None:
    
    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))


