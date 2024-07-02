import os
import sys
import json
import math
import argparse
from tqdm import tqdm

sys.path.append(".")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from utils.data.data_processor import PhotoChatDataProcessor as pcp
from utils.prompts import *

class LLaMaInferenceAgent:
    def __init__(self, src_path, dataset_name, model_name, task, decoding_param, device_map):
        # hyper-parameters
        self.src_path = src_path
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.task = task
        self.decoding_param = decoding_param
        self.device_map = device_map

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        )

        # models
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map=self.device_map,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        # dataset
        if self.dataset_name == "photochat":
            self.dataset = pcp().load_all_data(self.src_path, max_num=-1, key_first=True)
        else:
            raise NotImplementedError()
        
        if self.task == "query":
            self.get_prompt = get_descriptor_prompt
        elif self.task == "guess":
            self.get_prompt = get_guessing_prompt
        elif self.task == "sum":
            self.get_prompt = get_summarization_prompt
        else:
            raise("[!] invalid task type")
    
    def generate_text(self, prompt):
        toks = self.tokenizer(prompt, max_length=self.decoding_param["max_length"] - 1, truncation=True, return_tensors="pt")
        output = self.model.generate(**toks, **self.decoding_param)
        result = self.tokenizer.decode(output[0])

        return result
    
    def generate_text_for_split(self, split, seg=-1):
        n_data = len(self.dataset[split]["img_data"])
        seg_size = math.ceil(n_data / 4)

        if seg == -1:
            head, tail = 0, n_data
        else:
            head, tail = seg * seg_size, min((seg + 1) * seg_size, n_data)

        dataset = []
        pbar = tqdm(total=tail - head, ncols=0, desc=f"[*] llama for {split}")

        for i in range(head, tail):
            prompt = self.get_prompt(self.dataset[split]["diags"][i])
            reply = self.generate_text(prompt)

            dataset.append({
                "img_data": self.dataset[split]["img_data"][i],
                "img_desc": self.dataset[split]["img_desc"][i],
                "diags": self.dataset[split]["diags"][i],
                "prompt": prompt,
                "reply": reply,
            })

            pbar.update()

        pbar.close()

        return dataset

def main(args):
    # decoding_param = { "do_sample": args.do_sample, "top_k": args.top_k, "top_p": args.top_p, "max_length": args.max_length }
    decoding_param = { "max_length": args.max_length }

    llamaAgent = LLaMaInferenceAgent(
        args.src_path,
        args.dataset,
        args.model_name,
        args.task,
        decoding_param,
        args.device_map
    )

    for split in args.splits:
        dataset = llamaAgent.generate_text_for_split(split)

        filename = os.path.join(args.saved_path, f"{split}.json")
        with open(filename, "w", encoding="utf8") as writer:
            json.dump(dataset, writer, indent='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="photochat", choices=["photochat"])
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--saved_path", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="*", default=['test', 'train', 'dev'])
    parser.add_argument("--max_length", type=int, default=1600)

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--task", type=str, default="query", choices=["query", "guess", "sum"])

    # decoding algorithm
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path, exist_ok=False)
        print(f"[*] create saved path: {args.saved_path}")
    
    args.device_map = "auto"
    
    main(args)

