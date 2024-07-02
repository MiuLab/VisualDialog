import os
import sys
import clip
import json

sys.path.append(".")

from abc import ABC, abstractmethod
from PIL import Image
from utils.data.data_processor import PhotoChatDataProcessor as pcp
from utils.data.data_processor import PhotoChatLLaMaDataProcessor as pldp

class DataLoader(ABC):

    @abstractmethod
    def load_data(self, src_path):
        dataset = {}
        for split in self.splits:
            filename = os.path.join(src_path, f"{split}.json")
            with open(filename, "r", encoding="utf8") as reader:
                dataset[split] = json.load(reader)
            
        return dataset
    
    @abstractmethod
    def get_num_data(self, cur_set):
        return len(cur_set)
    
    @abstractmethod
    def none_reply(self, reply):
        return reply.lower() in ["none"]


class DialogueOnlyDataLoader(DataLoader):
    def load_data(self, src_path):
        dataset = pcp().load_all_data(src_path)
        return dataset
    
    def get_text(self, cur_set, idx):
        return "\n".join(cur_set[idx]["diags"])

    def get_clip_features(self, cur_set, idx, clip_processor, clip_model, device):
        text = self.get_text(cur_set, idx)
        
        image = clip_processor(Image.open(cur_set[idx]["img_data"])).unsqueeze(0).to(device)
        texts = clip.tokenize(
            [text], truncate=True
        ).to(device)
        img_desc = clip.tokenize(
            [cur_set[idx]["img_desc"]], truncate=True
        ).to(device)

        img_feat = clip_model.encode_image(image)
        text_feats = clip_model.encode_text(texts)
        desc_feat = clip_model.encode_text(img_desc)

        return img_feat, text_feats, desc_feat
    
    def get_num_data(self, cur_set):
        return super().get_num_data(cur_set)
    
    def none_reply(self, reply):
        return super().none_reply(reply)


class LLaMaDescriptorDataLoader(DataLoader):
    def __init__(self, queries=["main subject", "prominent objects in the foreground", "background scene", 
                                "events", "materials and attributes"]):
        self.queries = queries

    def load_data(self, src_path):
        dataset, _ = pldp().parse_all_json_data(src_path)
        return dataset
    
    def get_text(self, item):
        text = "\n".join(
            [
                f"the {key} of the photo is {val}".lower().replace("user 0", "a person").replace("user 1", "a person").replace("user", "a person")
                for key, val in item["reply"].items() if key in self.queries and not self.none_reply(val)
            ]
        )
        return text

    def get_clip_features(self, cur_set, idx, clip_processor, clip_model, device):
        item = cur_set[idx]

        image = clip_processor(Image.open(item["img_data"])).unsqueeze(0).to(device)

        text = self.get_text(item)
        texts = clip.tokenize([text] if not isinstance(text, list) else text, truncate=True).to(device)

        img_desc = item["img_desc"]
        img_desc = clip.tokenize([img_desc], truncate=True).to(device)

        img_feat = clip_model.encode_image(image)
        text_feats = clip_model.encode_text(texts)
        desc_feat = clip_model.encode_text(img_desc)

        return img_feat, text_feats, desc_feat
    
    def get_num_data(self, cur_set):
        return super().get_num_data(cur_set)
    
    def none_reply(self, reply):
        return super().none_reply(reply)


class LLaMaTaggedDataLoader(LLaMaDescriptorDataLoader):
    def load_data(self, src_path):
        dataset, _ = pldp().parse_all_tagged_data(src_path)
        return dataset
    
    def get_text(self, item):
        return item["reply"]

