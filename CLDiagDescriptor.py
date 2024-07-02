import os
import sys
import math
import time
import random
import argparse
import clip
from tqdm import tqdm

sys.path.append(".")

import torch
import torch.nn as nn
from utils.DNN import DNN
from utils.evaluate import reacall_at_k
from utils.data.data_loader import *

def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CLIPDescriptorModel(nn.Module):
    def __init__(self, clip_model_name, device, temperature=0.07, lambda_s=1.0):
        super(CLIPDescriptorModel, self).__init__()
        self.clip_model, self.clip_processor = clip.load(clip_model_name)
        self.clip_model.to(torch.float32)
        self.device = device

        self.lambda_s = nn.Parameter(torch.tensor([float(lambda_s)]))
        self.t = nn.Parameter(torch.tensor([float(temperature)]))
    
    def forward(self, norm_img_feats, norm_desc_feats, norm_text_feats, features="both"):
        """https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html"""
        """
        norm_img_feats: [n x 1 x 512] -> list
        norm_desc_feats: [n x 1 x 512] -> list
        norm_text_feats: [n x 512] -> list
        """
        norm_img_feats = torch.cat(norm_img_feats, dim=0)
        norm_desc_feats = torch.cat(norm_desc_feats, dim=0)
        norm_text_feats = torch.stack(norm_text_feats, dim=0)

        if features == "both":
            vision_nll = self.get_loss(norm_img_feats, norm_text_feats)
            scene_nll = self.get_loss(norm_desc_feats, norm_text_feats)
            loss = vision_nll * self.lambda_s.to(self.device) + scene_nll
        elif features == "img":
            vision_nll = self.get_loss(norm_img_feats, norm_text_feats)
            loss = vision_nll
        elif features == "text":
            scene_nll = self.get_loss(norm_desc_feats, norm_text_feats)
            loss = scene_nll
        else:
            raise("[!] not implement error")

        return loss
    
    def get_loss(self, norm_feats, norm_text_feats):
        sim_feats = torch.matmul(norm_feats, norm_text_feats.T)

        n_data = sim_feats.shape[0]

        self_mask = torch.eye(n_data, dtype=torch.bool, device=self.device)

        sim_feats = sim_feats / self.t.to(self.device)
        nll_i = -sim_feats[self_mask] + torch.logsumexp(sim_feats, dim=-1)
        nll_t = -sim_feats.T[self_mask] + torch.logsumexp(sim_feats.T, dim=-1)
        
        nll = (nll_i.mean() + nll_t.mean()) / 2

        return nll


class CLDiagDescriptorAgent:
    def __init__(self, src_path, clip_model_name, lr, batch_size, lambda_s, lambda_L, features, data_loader, device):
        # hyper-parameters
        self.src_path = src_path
        self.clip_model_name = clip_model_name
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_s = lambda_s
        self.lambda_L = lambda_L
        self.features = features
        self.data_loader = data_loader
        self.device = device

        # model
        self.clip_model = CLIPDescriptorModel(
            self.clip_model_name,
            self.device,
            lambda_s=self.lambda_s
        )

        self.set_train()
        self.optimizer = torch.optim.Adam(self.clip_model.parameters(), self.lr)

        # dataset
        self.dataset = self.data_loader.load_data(self.src_path)
    
    def set_train(self):
        self.clip_model.train()
    
    def set_eval(self):
        self.clip_model.eval()

    def train(self, n_epochs, saved_path, ks=[1, 5, 10], test_epoch=5, save_model=True):
        record_name = time.strftime("%Y%b%d-%H-%M-%S", time.gmtime())
        model_path = os.path.join(saved_path, f"{record_name}.ckpt")
        max_score = 0

        for epoch in range(n_epochs):
            print(f"[*] epoch {epoch + 1}")
            self.train_one_epoch()

            valid_recalls = self.validation(ks)
            valid_score = sum(valid_recalls)
            for k, recalls in zip(ks, valid_recalls):
                print(f"[*] valid r@{k}: {recalls:.4f}")
            print(f"[*] valid score: {valid_score:.4f}")

            if valid_score > max_score:
                max_score = valid_score
                print("[*] save model")
                if save_model:
                    torch.save(self.clip_model.state_dict(), model_path)

            if epoch % test_epoch == test_epoch - 1 or epoch == n_epochs - 1:
                test_recalls = self.test()
                test_score = sum(test_recalls)
                for k, recalls in zip(ks, test_recalls):
                    print(f"[*] test r@{k}: {recalls:.4f}")
                print(f"[*] test score: {test_score:.4f}")
        
        if save_model:
            test_recalls = self.test(model_path=model_path)
            test_score = sum(test_recalls)
            for k, recalls in zip(ks, test_recalls):
                print(f"[*] test r@{k}: {recalls:.4f}")
            print(f"[*] best test score: {test_score:.4f}")

    def train_one_epoch(self):
        train_set = self.dataset['train']
        n_data = self.data_loader.get_num_data(train_set)

        self.set_train()

        steps = list(range(n_data))
        random.shuffle(steps)
        n_steps = int(math.ceil(n_data / self.batch_size))

        pbar = tqdm(total=n_steps, ncols=0, desc="[*] descriptor model training")

        for step in range(n_steps):
            head = step * self.batch_size
            tail = min((step + 1) * self.batch_size, n_data)

            self.optimizer.zero_grad()

            norm_img_feats, norm_desc_feats, norm_text_feats = [], [], []
            for i in steps[head:tail]:
                img_feat, text_feats, desc_feat = self.data_loader.get_clip_features(
                    train_set,
                    i,
                    self.clip_model.clip_processor,
                    self.clip_model.clip_model,
                    self.device
                )

                # remove all the zero rows
                non_zero_rows = torch.any(text_feats != 0, dim=1)
                text_feats = text_feats[non_zero_rows]

                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                desc_feat = desc_feat / desc_feat.norm(dim=-1, keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                if text_feats.shape[0]:
                    norm_img_feats.append(img_feat)
                    norm_desc_feats.append(desc_feat)
                    norm_text_feats.append(text_feats[0])

            loss = self.clip_model(norm_img_feats, norm_desc_feats, norm_text_feats, features=self.features)

            loss.backward()
            self.optimizer.step()

            pbar.set_postfix({"loss": loss.item(), "t": f"{self.clip_model.t.item():.4f}", "lambda": f"{self.clip_model.lambda_s.item():.4f}"})

            pbar.update()
        pbar.close()
    
    def validation(self, ks=[1, 5, 10]):
        valid_set = self.dataset["dev"]
        n_data = self.data_loader.get_num_data(valid_set)

        self.set_eval()

        recalls = [0 for _ in ks]
        n_steps = int(math.ceil(n_data / self.batch_size))

        pbar = tqdm(total=n_steps, ncols=0, desc="[*] descriptor model validation")

        for step in range(n_steps):
            head = step * self.batch_size
            tail = min((step + 1) * self.batch_size, n_data)

            norm_img_feats, norm_desc_feats, norm_text_feats = [], [], []
            for i in range(head, tail):
                with torch.no_grad():
                    img_feat, text_feats, desc_feat = self.data_loader.get_clip_features(
                        valid_set,
                        i,
                        self.clip_model.clip_processor,
                        self.clip_model.clip_model,
                        self.device
                    )

                # remove all the zero rows
                non_zero_rows = torch.any(text_feats != 0, dim=1)
                text_feats = text_feats[non_zero_rows]

                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                desc_feat = desc_feat / desc_feat.norm(dim=-1, keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                if text_feats.shape[0]:
                    norm_img_feats.append(img_feat)
                    norm_desc_feats.append(desc_feat)
                    norm_text_feats.append(text_feats)
            
            norm_img_feats = torch.stack(norm_img_feats, dim=0).squeeze(1).to(self.device)
            norm_desc_feats = torch.stack(norm_desc_feats, dim=0).squeeze(1).to(self.device)

            n_data_batch = norm_img_feats.shape[0]

            score_mat = torch.empty([n_data_batch, n_data_batch]).to(self.device)
            
            for i in range(n_data_batch):
                text_feats = norm_text_feats[i].to(self.device)

                if self.features == "both":
                    img_scores = torch.matmul(norm_img_feats, text_feats.T)
                    img_scores = torch.mean(img_scores, dim=-1)

                    text_scores = torch.matmul(norm_desc_feats, text_feats.T)
                    text_scores = torch.mean(text_scores, dim=-1)

                    score_mat[i] = torch.add(img_scores * self.clip_model.lambda_s.to(self.device), text_scores)
                elif self.features == "img":
                    img_scores = torch.matmul(norm_img_feats, text_feats.T)
                    img_scores = torch.mean(img_scores, dim=-1)

                    score_mat[i] = img_scores
                elif self.features == "text":
                    text_scores = torch.matmul(norm_desc_feats, text_feats.T)
                    text_scores = torch.mean(text_scores, dim=-1)

                    score_mat[i] = text_scores
                else:
                    raise("[!] not implement error")
            
            labels = torch.LongTensor(torch.arange(n_data_batch)).to(self.device)

            recalls = [recalls[j] + reacall_at_k(score_mat, labels, ks[j]) for j in range(len(ks))]

            pbar.update()
        pbar.close()

        recalls = [k / n_data for k in recalls]
        
        return recalls
    
    def test(self, model_path=None, ks=[1, 5, 10], record=False):
        if model_path is None:
            clip_model = self.clip_model
        else:
            clip_model = CLIPDescriptorModel(self.clip_model_name, self.device)
            clip_model.load_state_dict(torch.load(model_path, map_location=self.device))

        test_set = self.dataset['test']
        n_data = self.data_loader.get_num_data(test_set)

        pbar = tqdm(total=n_data, ncols=0, desc=f"[*] get clip features from test set")

        self.set_eval()

        norm_img_feats, norm_desc_feats, norm_text_feats = [], [], []
        for i in range(n_data):
            with torch.no_grad():
                img_feat, text_feats, desc_feat = self.data_loader.get_clip_features(
                    test_set,
                    i,
                    clip_model.clip_processor,
                    clip_model.clip_model,
                    self.device
                )

            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            desc_feat /= desc_feat.norm(dim=-1, keepdim=True)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)

            norm_img_feats.append(img_feat)
            norm_desc_feats.append(desc_feat)
            norm_text_feats.append(text_feats)

            pbar.update()
        pbar.close()
        
        recall_at_ks = [0 for _ in range(len(ks))]

        norm_img_feats = torch.stack(norm_img_feats, dim=0).squeeze(1).to(self.device)
        norm_desc_feats = torch.stack(norm_desc_feats, dim=0).squeeze(1).to(self.device)
        
        pbar = tqdm(total=n_data, ncols=0, desc="[*] descriptor model testing")
        records = []

        for i in range(n_data):
            text_feats = norm_text_feats[i].to(self.device)

            if self.features == "both":
                img_scores = torch.matmul(norm_img_feats, text_feats.T)
                img_scores = torch.mean(img_scores, dim=-1)

                text_scores = torch.matmul(norm_desc_feats, text_feats.T)
                text_scores = torch.mean(text_scores, dim=-1)

                scores = img_scores * self.clip_model.lambda_s.to(self.device) + text_scores
            elif self.features == "img":
                img_scores = torch.matmul(norm_img_feats, text_feats.T)
                img_scores = torch.mean(img_scores, dim=-1)

                scores = img_scores
            elif self.features == "text":
                text_scores = torch.matmul(norm_desc_feats, text_feats.T)
                text_scores = torch.mean(text_scores, dim=-1)

                scores = text_scores
            else:
                raise("[!] not implement error")

            _, indices = torch.sort(scores, descending=True)
            for idx, k in enumerate(ks):
                if i in indices[:k]:
                    recall_at_ks[idx] += 1

            pbar.update()
        pbar.close()

        recall_at_ks = [k / n_data for k in recall_at_ks]

        if record:
            return recall_at_ks, records
        
        return recall_at_ks
            
    def zero_shot_inference(self, ks=[1, 5, 10]):
        test_set = self.dataset['test']
        n_data = self.data_loader.get_num_data(test_set)

        self.set_eval()

        pbar = tqdm(total=n_data, ncols=0, desc=f"[*] get clip features from test set")

        norm_img_feats, norm_desc_feats, norm_text_feats = [], [], []
        for i in range(n_data):
            with torch.no_grad():
                img_feat, text_feats, desc_feat = self.data_loader.get_clip_features(
                    test_set,
                    i,
                    self.clip_model.clip_processor,
                    self.clip_model.clip_model,
                    self.device
                )

            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            desc_feat /= desc_feat.norm(dim=-1, keepdim=True)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)

            norm_img_feats.append(img_feat)
            norm_desc_feats.append(desc_feat)
            norm_text_feats.append(text_feats)

            pbar.update()
        pbar.close()
        
        recall_at_ks = [0 for _ in range(len(ks))]

        norm_img_feats = torch.stack(norm_img_feats, dim=0).squeeze(1).to(self.device)
        norm_desc_feats = torch.stack(norm_desc_feats, dim=0).squeeze(1).to(self.device)
        
        pbar = tqdm(total=n_data, ncols=0, desc="[*] zero shot inference")

        for i in range(n_data):
            text_feats = norm_text_feats[i].to(self.device)

            if self.features == "both":
                img_scores = torch.matmul(norm_img_feats, text_feats.T)
                img_scores = torch.mean(img_scores, dim=-1)

                text_scores = torch.matmul(norm_desc_feats, text_feats.T)
                text_scores = torch.mean(text_scores, dim=-1)

                scores = torch.add(img_scores * self.lambda_s, text_scores)
            elif self.features == "img":
                img_scores = torch.matmul(norm_img_feats, text_feats.T)
                img_scores = torch.mean(img_scores, dim=-1)

                scores = img_scores
            elif self.features == "text":
                text_scores = torch.matmul(norm_desc_feats, text_feats.T)
                text_scores = torch.mean(text_scores, dim=-1)

                scores = text_scores
            else:
                raise("[!] not implement error")

            _, indices = torch.sort(scores, descending=True)
            for idx, k in enumerate(ks):
                if i in indices[:k]:
                    recall_at_ks[idx] += 1

            pbar.update()
        pbar.close()
        
        recall_at_ks = [k / n_data for k in recall_at_ks]
        scores = sum(recall_at_ks)
        for i in range(len(recall_at_ks)):
            print(f"[*] R@{ks[i]}: {recall_at_ks[i]:.4f}")
        print(f"[*] test score: {scores:.4f}")

        return


def main(args):
    if args.task in ["photochat-diag-only"]:
        data_loader = DialogueOnlyDataLoader()
    elif args.task in ["photochat-llama13b-query", "photochat-llama7b-query"]:
        data_loader = LLaMaDescriptorDataLoader()
    elif args.task in ["photochat-llama13b-sum", "photochat-llama7b-sum",
                       "photochat-llama13b-guess", "photochat-llama7b-guess"]:
        data_loader = LLaMaTaggedDataLoader()
    else:
        raise("[!] task not specified")
    
    DDA = CLDiagDescriptorAgent(
        args.src_path,
        args.clip_model_name,
        args.lr,
        args.batch_size,
        args.lambda_s,
        args.lambda_L,
        args.features,
        data_loader,
        args.device
    )

    if args.zero_shot:
        DDA.zero_shot_inference()
    else:
        DDA.train(args.n_epochs, args.saved_path, test_epoch=args.test_epoch, save_model=args.save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=[
        "photochat-diag-only", "photochat-llama13b-query", "photochat-llama7b-query",
        "photochat-llama13b-sum", "photochat-llama7b-sum", "photochat-llama13b-guess", "photochat-llama7b-guess"
    ])
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--saved_path", type=str, default="./saved_model")

    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--test_epoch", type=int, default=1)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--lambda_L", type=float, default=1.0)
    parser.add_argument("--features", type=str, choices=['img', 'text', 'both'], default='both')
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--zero_shot", action="store_true")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {args.device}")

    if not args.save_model:
        print("won't save model")

    assert(os.path.exists(args.src_path)), f"[!] src path {args.src_path} doesn't exist"
    if not os.path.exists(args.saved_path):
        os.mkdir(args.saved_path)
        print(f"[! create saved dir: {args.saved_path}")

    setup_seed(args.seed)

    main(args)