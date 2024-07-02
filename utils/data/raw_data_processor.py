import os
import glob
import json
import argparse
import requests

class PhotoChatRawDataProcessor:
    def __init__(self, raw_path, saved_path):
        self.raw_path = raw_path
        self.saved_path = saved_path
        self.image_pool = os.path.join(self.saved_path, "images")

        if not os.path.exists(self.image_pool):
            os.mkdir(self.image_pool)
            print(f"[*] create image pool: {self.image_pool}")
    
    def crawl_image_with_url(self, url):
        response = requests.get(url)
        return response
    
    def process_raw_data(self):
        """Process all raw data"""
        self.process_raw_data_in_split("train")
        self.process_raw_data_in_split("dev")
        self.process_raw_data_in_split("test")

        return

    def process_raw_data_in_split(self, split="train"):
        """Input:
        split: can be 'train', 'dev', or 'test'
        """
        assert(split in ["train", "dev", "test"]), f"[!] split '{split}' is not in 'train', 'dev', or 'test'"

        dir_path = os.path.join(self.raw_path, split)
        filenames = sorted(glob.glob(os.path.join(dir_path, "*.json")))

        all_data = []

        for filename in filenames:
            with open(filename, "r", encoding="utf8") as reader:
                raw_data = json.load(reader)

            self.collect_images_in_json(filename)

            for data in raw_data:
                photo_path, photo_id = data["photo_id"].split('/')
                photo_path = os.path.join(self.image_pool, photo_path, f"{photo_id}.jpg")
                
                if os.path.exists(photo_path):
                    all_data.append(data)
        
        with open(os.path.join(self.saved_path, f"{split}.json"), "w", encoding="utf8") as writer:
            json.dump(all_data, writer, indent='\t')
        
        print(f"[*] total {len(all_data)} data")

        return
    
    def collect_images_in_json(self, filename):
        """Collect images in a json file"""
        with open(filename, "r", encoding="utf8") as reader:
            raw_data = json.load(reader)
        
        error_count = 0

        err_writer = open(f"{filename}-err_log.txt", "w", encoding="utf8")
        
        for data in raw_data:
            photo_url = data["photo_url"]
            photo_path, photo_id = data["photo_id"].split('/')
            photo_path = os.path.join(self.image_pool, photo_path)

            if not os.path.exists(photo_path):
                os.mkdir(photo_path)
                print(f"[*] create image subdir: {photo_path}")
            
            photo_path = os.path.join(photo_path, f"{photo_id}.jpg")
            if os.path.exists(photo_path):
                continue

            response = self.crawl_image_with_url(photo_url)

            if not response.status_code == 200:
                print(f"[!] fail to crawl url {photo_url}, status code: {response.status_code}")
                print(f"\t{data['photo_id']}")
                err_writer.write(f"{data['photo_id']}\n")
                error_count += 1
                continue

            with open(photo_path, "wb") as writer:
                writer.write(response.content)
        
        err_writer.close()
        
        print(f"[!] {error_count} error in {filename}")

        return

def main(args):
    p = PhotoChatRawDataProcessor(args.raw_path, args.saved_path)
    p.process_raw_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str)
    parser.add_argument("--saved_path", type=str)

    args = parser.parse_args()

    assert(os.path.exists(args.raw_path)), f"Raw path {args.raw_path} not found"
    print(f"[*] raw path: {args.raw_path}")

    if os.path.exists(args.saved_path):
        print(f"[*] saved path: {args.saved_path} already exists")
    else:
        os.mkdir(args.saved_path)
        print(f"[*] create saved path: {args.saved_path}")

    main(args)