import os
import re
import json
from abc import ABC, abstractmethod
from tqdm import tqdm

NO_ERROR = 0
JSON_FORMAT_ERROR = -1
KEY_ERROR = -2
ERRORS = { -1: "JSON_FORMAT_ERROR", -2: "KEY_ERROR" }

class DataProcessor(ABC):
    """data processor for loading and processing data"""

    @abstractmethod
    def load_all_data(self, src_path, max_num=-1, key_first=False):
        """
        Load data for all the splits
        Output:
        { "train": list(dict), "test": list(dict), ... } if key_first = False
        { "train": dict(str: list) } if key_first = True
        """
        dataset = {}
        for split in self.splits:
            dataset[split] = self.load_data(src_path=src_path, split=split, max_num=max_num, key_first=key_first)
        return dataset
    
    def load_data(self, src_path, split, max_num=-1, key_first=False):
        """
        Load data for single split
        Output:
        list(dict) if key_first = False
        dict(str: list) if key_first = True
        """
        pass


class LLaMaDataProcessor(ABC):
    """data processor for loading and processing llama data"""

    @abstractmethod
    def parse_all_json_data(self, src_path):
        dataset, failed_set = {}, {}
        for split in self.splits:
            data, failed = self.parse_json_data(src_path, split)
            dataset[split] = data
            failed_set[split] = failed
        
        return dataset, failed_set
    
    @abstractmethod
    def parse_json_data(self, src_path, split):
        pass

    @abstractmethod
    def parse_all_tagged_data(self, src_path):
        dataset, failed_set = {}, {}
        for split in self.splits:
            data, failed = self.parse_tagged_data(src_path, split)
            dataset[split] = data
            failed_set[split] = failed
        
        return dataset, failed_set
    
    @abstractmethod
    def parse_tagged_data(self, src_path, split):
        pass


class PhotoChatDataProcessor(DataProcessor):
    """Prcocessor for loading PhotoChat data"""
    def __init__(self):
        self.splits = ["train", "dev", "test"]

    def load_all_data(self, src_path, max_num=-1, key_first=False):
        return super().load_all_data(src_path, max_num, key_first)
    
    def load_data(self, src_path, split, max_num=-1, key_first=False):
        """
        keys:
        "img_data": str
        "img_desc": str
        "diags": list(str)
        """
        filename = os.path.join(src_path, f"{split}.json")
        with open(filename, "r", encoding="utf8") as reader:
            all_data = json.load(reader)
        
        image_pool = os.path.join(src_path, "images")

        max_num = len(all_data) if max_num == -1 else min(max_num, len(all_data))
        
        parsed_data = []
        for data in tqdm(all_data, desc=f"[*] Load PhotoChat {split}"):
            parsed_data.append({
                "diags": self.parse_raw_dialogue(data["dialogue"]),
                "img_data": self.parse_image_data(data["photo_id"], image_pool),
                "img_desc": data["photo_description"]
            })
        
        if key_first:
            parsed_data = {
                "diags": [data["diags"] for data in parsed_data],
                "img_data": [data["img_data"] for data in parsed_data],
                "img_desc": [data["img_desc"] for data in parsed_data],
            }
        return parsed_data
    
    def parse_raw_dialogue(self, dialogue):
        dialogues = []
        for turn in dialogue:
            if turn["share_photo"]:
                break
            dialogues.append(f"User {turn['user_id']}: " + turn["message"])
        return dialogues
    
    def parse_image_data(self, photo_id, image_pool):
        split, idx = photo_id.split('/')
        return os.path.join(image_pool, split, f"{idx}.jpg")


class PhotoChatLLaMaDataProcessor(LLaMaDataProcessor):
    """Processor for photochat llama-generated data"""
    def __init__(self):
        self.splits = ["train", "test", "dev"]

    def get_keys_for_json_format_prompts(self, reply):
        lines = reply.split('\n')
        key_lines = []

        for line in lines:
            if line.startswith("- "):
                key_lines.append(line)
            elif len(key_lines):
                break
        
        keys = [line.strip().split(':')[0].replace("- ", "") for line in key_lines]

        return keys

    def parse_all_json_data(self, src_path):
        return super().parse_all_json_data(src_path)
    
    def parse_json_data(self, src_path, split):
        filename = os.path.join(src_path, f"{split}.json")
        with open(filename, "r", encoding="utf8") as reader:
            all_data = json.load(reader)
        
        dataset, errors = [], { "JSON_FORMAT_ERROR": 0, "KEY_ERROR": 0 }
        keys = self.get_keys_for_json_format_prompts(all_data[-1]["reply"])

        for data in all_data:
            status, descs = self.parse_json_reply(data, keys)

            if status == NO_ERROR:
                res = data.copy()
                res["reply"] = descs
                dataset.append(res)
            else:
                errors[ERRORS[status]] += 1
                print(descs)
                print(data["img_data"], ERRORS[status])
        
        print(f"[*] parse errors: {errors}")
        
        return dataset, errors

    def parse_json_reply(self, data, keys):
        reply = data["reply"].split("\n\nAnswers:\n\n")[-1].strip().replace("</s>", "")
        reply = self.correct_keys(reply)
        reply = self.merge_multiple_json(reply)
        reply = self.remove_comments(reply)
        reply = self.merge_separate_quotes(reply)
        
        try:
            json_data = json.loads(reply)
        except Exception as e:
            print(e)
            return JSON_FORMAT_ERROR, reply
        
        if list(json_data.keys()) != keys:
            return KEY_ERROR, json.dumps(json_data)
        
        for key, val in json_data.items():
            if val is None:
                json_data[key] = "none"
            elif isinstance(val, list):
                json_data[key] = ", ".join(val)
        
        return NO_ERROR, json_data
    
    def correct_keys(self, reply):
        reply = reply.replace("prominent objects in foreground", "prominent objects in the foreground")
        return reply
    
    def remove_comments(self, reply):
        reply = reply.split("}")[0].strip() + "\n}"
        return reply

    def merge_separate_quotes(self, reply):
        reply = reply.replace("\", \"", ", ").replace("\"\n\"", "\",\n\"").replace("\",\n}", "\"\n}")
        return reply
    
    def correct_key_format(self, reply, keys):
        for key in keys:
            reply = reply.replace(" " + key + "\"", " \"" + key + "\"")
        return reply
    
    def merge_multiple_json(self, reply):
        return reply.replace(" }\n{ ", ",\n")
    
    def parse_all_tagged_data(self, src_path):
        return super().parse_all_tagged_data(src_path)

    def parse_tagged_data(self, src_path, split):
        filename = os.path.join(src_path, f"{split}.json")
        with open(filename, "r", encoding="utf8") as reader:
            all_data = json.load(reader)
        
        dataset, failed = [], []
        errors = {}

        for data in all_data:
            reply = self.parse_tagged_reply(data["reply"])
            if isinstance(reply, int):
                if not reply in errors:
                    errors[reply] = 0
                errors[reply] += 1
                failed.append(data)
                print(data["reply"])
                print(data["img_data"], reply)
            else:
                res = data.copy()
                res["reply"] = reply
                dataset.append(res)

        print(f"[*] parse error: {errors}")

        return dataset, failed

    def parse_tagged_reply(self, reply):
        pattern = r'<start of answer>(.*?)(<end of answer>|</end of answer>)'

        matches = re.findall(pattern, reply, re.DOTALL)

        if len(matches) < 1:
            return -1
        
        return matches[0][0].strip()
