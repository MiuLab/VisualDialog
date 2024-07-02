import sys

sys.path.append(".")

import os
import json
import glob
from tqdm import tqdm
from ultralytics import YOLO

def get_image_id_from_filename(image_path):
    filename = image_path.split("/")[-1]
    filename = filename.replace(".jpg", "").strip()
    return filename

class Yolov8ObjectDetector:
    def __init__(self, model_path="yolov8x", conf=0.25):
        self.model_path = model_path
        self.conf = conf

        self.model = YOLO(self.model_path)

    def predict_for_dataset(self, dataset):
        for idx in tqdm(range(len(dataset))):
            detection = self.predict(dataset[idx]["img_data"])
            results = self.postprocess(detection)
            dataset[idx][self.key_name] = results
        return dataset
    
    def predict(self, img_path):
        result = self.model(source=img_path, conf=self.conf, save=False, verbose=False)
        return result
    
    def postprocess(self, pred):
        cls = pred[0].boxes.cls.detach().cpu().tolist()
        names = pred[0].names
        return [names[int(c)] for c in cls]

    def get_image_repr_for_directory(self, dir_path):
        image_paths = glob.glob(os.path.join(dir_path, "*.jpg"))
        captions = {}

        for image_path in tqdm(image_paths):
            image_id = get_image_id_from_filename(image_path)
            
            try:
                detections = self.model(source=image_path, conf=self.conf, save=False, verbose=False)
                results = self.postprocess(detections)
                captions[image_id] = ", ".join(results)
            except:
                print("[!] No")
                continue
        
        with open(os.path.join(dir_path, "yolov8_repr.json"), "w", encoding="utf8") as writer:
            json.dump(captions, writer, indent='\t')
        
        return

if __name__ == "__main__":
    agent = Yolov8ObjectDetector()
    agent.get_image_repr_for_directory("VisDial/data/data/VisualDialog_val2018")
