from html import entities
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import clip
import matplotlib.pyplot as plt
import json
from collections import Counter
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn import cluster
from sklearn.preprocessing import QuantileTransformer
import networkx as nx
from utils import *
from sklearn.cluster import DBSCAN
import argparse

NUM_ROWS_EACH_PAGE = 25


class VisualLinguisticRegularity:
    """
    A class identifying hateful meme variants given a hateful meme, e.g., Happy Merchant, and a list of texts, by extracting visual-linguistic semantic regularity.

    Visual-linguistic semantic regularity with both the image and text embeddings: 
        
        *** original meme + influencer (text) -> variant ***

    Parameters:
    ----------
    topk: how many variants to identify for a pair of original meme and a textual influencer
    entity_topk: top-k entities in each type, e.g., People
    model_file: fine-tuned CLIP model
    entity_dir: files of entity list
    image_embeddings: all image embeddings
    image_dict: a file of all images locations/urls and phash
    image_root: root directory of saving images
    data_file: 4chan data, containing textal post and image locations
    save_dir: directory to save results

    Functions
    ----------
    retrieve_variants: retrieve variants given a hateful meme and a list of entities
    temporal_analysis: calculate the occurrence of variants and select the popular one between top-2s

    Outputs
    ----------
    variant ids, entities in .npz
    variant visualization
    variant occurrences
    

    """

    def __init__(self, 
                topk=3, 
                entity_topk=20, 
                model_file=None,
                entity_dir=None,
                image_embeddings=None, 
                image_dict=None,
                image_root=None,
                data_file=None,
                save_dir=None):

        self.model_file = model_file
        self.topk = topk
        self.entity_topk = entity_topk
        self.image_root = image_root
        self.entity_dir = entity_dir
        self.base_dir  = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.base_dir, save_dir)

        self.image_embeddings = torch.from_numpy(np.load(image_embeddings)).type(torch.float64)
        self.image_norm = self.image_embeddings / torch.norm(self.image_embeddings, dim=-1, keepdim=True)

        with open(image_dict, "r") as file:
            self.image_dictionary = file.readlines()
            file.close()

        with open(data_file, "r") as file1:
            self.all_posts = file1.readlines()
            file1.close()
            
        self.entity_dict = {}
        for en_type in ["people", "gpe", "norp", "org"]:
            file_lst = open(f"{entity_dir}/entities_{en_type}.txt", "r").readlines()
            self.entity_dict[en_type] = [file_lst[idx].split()[0] for idx in range(self.entity_topk)]

        self.model, _ = clip.load(self.model_file, device=device)
        self.model = self.model.eval()

    def retrieve_variants(self, img_idx=None, entity_type=None):
   
        with torch.no_grad():
            variant_indices = []
            variant_sim = []

            entities = self.entity_dict[entity_type]
            for entity in entities:
                text = clip.tokenize(entity).to(device)
                _text_embedding = self.model.encode_text(text).type(torch.float64).squeeze()
                norm_text = _text_embedding / torch.norm(_text_embedding, keepdim=True)
                candidate_embedding = 0.2*self.image_norm[img_idx].to(device) + 0.8*norm_text # this numerical relation is determined with experiments
                candidate_embedding /= torch.norm(candidate_embedding, keepdim=True)

                similarities = torch.matmul(candidate_embedding, self.image_norm.T.to(device))
                _topk_similarities, _topk_indices = torch.topk(similarities, self.topk, dim=-1) # i.e., [3], [3]
                _topk_similarities, _topk_indices = _topk_similarities.detach().cpu().numpy(), _topk_indices.detach().cpu().numpy()
                variant_indices.append(_topk_indices)
                # variant_sim.append(_topk_similarities)

        self.image_norm.detach().cpu()

        return entities, np.array(variant_indices)

    def visualize_variants(self, img_idx, show_graphes=True):

        for en_type in ["people", "gpe", "norp", "org"]:

            entities, variant_indices = self.retrieve_variants(img_idx=img_idx, entity_type=en_type)            
            print(f"Now {variant_indices.shape[0]} pairs in total")
            np.savez(f"{self.save_dir}/variants_{en_type}.npz", entity=entities, variant=variant_indices)

            def _draw_figuer_by_page(page, entities, variant_ids):
                plt.figure(figsize=(100, 100))
                for row, (entity, v_id) in enumerate(zip(entities, variant_ids)):
                    image_in_row = v_id
                    for col, idx in enumerate(image_in_row):
                        plt.subplot(self.entity_topk, self.topk, (row * len(image_in_row) + col)+1)
                        _, image = self.visualize_meme(int(idx))
                        plt.imshow(image)
                        plt.xticks([])
                        plt.yticks([])
                        plt.xlabel(f"{entity} Top-{col+1}", fontsize=24)

                plt.savefig(f"{self.save_dir}/variants_{en_type}_{page}.pdf")
                plt.close()

            if show_graphes:
                num_pages = variant_indices.shape[0] // NUM_ROWS_EACH_PAGE + 1
                for i in range(num_pages):
                    start = i*NUM_ROWS_EACH_PAGE
                    end = min((i+1)*NUM_ROWS_EACH_PAGE, variant_indices.shape[0])
                    entities_lst, variant_ids = entities[start:end], variant_indices[start:end]
                    _draw_figuer_by_page(i, entities_lst, variant_ids)          
            
    def draw_topks(self, top1=True):
         for en_type in ["people", "gpe", "norp", "org"]:
             res = np.load(f"{self.save_dir}/variants_{en_type}.npz")
             entities, variant_indices = res["entity"], res["variant"]

             for i in range(variant_indices.shape[0]):
                if top1:
                    select_indices = variant_indices[i][0]
                else:
                    select_indices = variant_indices[i][1]

                self.visualize_pairs(i, en_type, entities[i], select_indices)

    def visualize_meme(self, idx):
        line = json.loads(self.image_dictionary[idx])
        image_dir = os.path.join(self.image_root, line["img_loc"].lstrip("/"))
        image = Image.open(image_dir).convert("RGB")
        return line, image
    
    def idx2phash(self, idx):
        return json.loads(self.image_dictionary[idx])["phash"]

    def visualize_pairs(self, i, en_type, entity, variant_id, tgt_path=None):

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        _, img = self.visualize_meme(variant_id)
        ax.imshow(img)
        ax.axis("off")

        if ((en_type == "org") & (len(entity) <=4)):
            title =  entity.upper()
        else:
            if entity == "uk":
                title =  entity.upper()
            else:
                title =  entity.title()

        ax.set_title(title, fontsize=200)

        plt.subplots_adjust()
        tgt_path = f"{self.save_dir}/{en_type}"
        Path(tgt_path).mkdir(exist_ok=True, parents=True)
        # plt.savefig(f"{tgt_path}/{i}.png", bbox_inches="tight")
        plt.savefig(f"{tgt_path}/{i}_{entity}_figure_{variant_id}.png", bbox_inches="tight")
        plt.close(fig)
    
    def test_phash(self, idx_0, idx_1):
        return self.idx2phash(idx_0) == self.idx2phash(idx_1)

    def _compute_occurrences(self, img_idx): # img_idx in image_dictionary
        from datetime import datetime, date, timedelta

        center_phash = json.loads(self.image_dictionary[img_idx])["phash"]
        img_locs = []
        for line in self.image_dictionary:
            line = json.loads(line)
            phash = line["phash"]
            if phash == center_phash:
                img_locs.append(line["img_loc"].lstrip("/"))

        timestamps = []
        # find all the posts that contained on of the above images
        for line in self.all_posts:
            line = json.loads(line)
            img_loc = line["img_loc"]
            time = int(line["time"])
            if img_loc in img_locs:
                timestamps.append(time)

        timestamps = sorted(timestamps)
        dates = [datetime.fromtimestamp(time).strftime("%y-%m-%d") for time in timestamps]
        df = pd.DataFrame(data=dates, columns=["date"])
        count = df.groupby("date")["date"].count()
        return count

    def compute_occurrences(self, en_type, top1=True):
        from datetime import datetime, date, timedelta

        res = np.load(f"{self.save_dir}/variants_{en_type}.npz")
        entities, variants= res["entity"], res["variant"]
        if top1:
            variants = variants[:,0] # top 1st variant
        else:
            variants = variants[:,1] # top 2nd variant
        
        # create an empty dataframe
        date_list = []
        startdate = date(2016, 7, 1)
        while startdate <= date(2017, 7, 30):
            startdate += timedelta(days=1)
            date_list.append(startdate.strftime("%y-%m-%d"))
            
        df = pd.DataFrame([])
        df.index = date_list
        df_copy = df.copy()
        for entity, v_id in zip(entities, variants):
            occurrence_count = self._compute_occurrences(img_idx=v_id)
            df[entity] = pd.merge(df_copy, occurrence_count, left_index=True, right_index=True)
            print(f"{entity} variant, occurrences: {len(occurrence_count)}")
        df = df.fillna(0)
        return df
        
    def temporal_analysis(self):
        for en_type in ["people", "gpe", "norp", "org"]:
            top1_df = self.compute_occurrences(en_type, top1=True)
            top2_df = self.compute_occurrences(en_type, top1=False)
            res = np.load(f"{self.save_dir}/variants_{en_type}.npz")
            entities, variants= res["entity"], res["variant"]
            top1_id, top2_id = variants[:,0], variants[:,1]
            
            new_df = pd.DataFrame([])
            most_popular_ids = []
            for i, col in enumerate(top1_df.columns): # 
                _top1_df = top1_df.loc["16-07-01":"17-07-30", col]
                _top2_df = top2_df.loc["16-07-01":"17-07-30", col]
                if _top1_df.sum() >= _top2_df.sum():
                    new_df = pd.concat([new_df, _top1_df], axis=1)
                    most_popular_ids.append(top1_id[i])
                else:
                    new_df = pd.concat([new_df, _top2_df], axis=1)
                    most_popular_ids.append(top2_id[i])
        
            new_df.to_csv(f"{self.save_dir}/occurrence_{en_type}.csv")
            for i, id in enumerate(most_popular_ids):
                entity = entities[i]
                ME.visualize_pairs(i, en_type, entity, id, os.path.join(f"{self.save_dir}", f"{en_type}"))

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--meme", 
        choices=["HappyMerchant", "PepeTheFrog"], 
        help="meme name, choose from HappyMerchant and PepeTheFrog", 
        default=None
    )
    parser.add_argument(
        "--model_file",
        type=str,
        nargs="?",
        default="model_checkpoint/clip/lr_1e-06/model_60000.pt",
        help="CLIP model path, load from the fine-tuned checkpoint"
    )
    parser.add_argument(
        "--entity_dir", 
        type=str, 
        help="four types of entities", 
        default="data/entities"
    )
    parser.add_argument(
        "--image_embeddings", 
        type=str, 
        help="image embedding file", 
        default=None
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        help="4chan post file containing all image locations and text posts", 
        default=None
    )
    parser.add_argument(
        "--image_dict", 
        type=str, 
        help="image dictionary saving image locations and phash", 
        default=None
    )
    parser.add_argument(
        "--image_root", 
        type=str, 
        help="image saving directory", 
        default=None
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        help="saving directory", 
        default="result/visual_linguistic_regularity"
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="?",
        default="cuda",
        help="device"
    )

    CFG = parser.parse_args()
    CFG.save_dir = os.path.join(CFG.save_dir, CFG.meme)
    Path(CFG.save_dir).mkdir(exist_ok=True, parents=True)
    device = CFG.device
    # memes_index = {"HappyMerchant": 738425, "PepeTheFrog": 4465}
    memes_index = {"HappyMerchant": 0, "PepeTheFrog": 1} # wrong index, only for code testing

    ME = VisualLinguisticRegularity(topk=3,
                                    entity_topk=20, 
                                    model_file=CFG.model_file,
                                    entity_dir=CFG.entity_dir,
                                    image_embeddings=CFG.image_embeddings, 
                                    data_file=CFG.data_file,
                                    image_dict=CFG.image_dict,
                                    image_root=CFG.image_root,
                                    save_dir=CFG.save_dir)

    ME.visualize_variants(img_idx=memes_index[CFG.meme], show_graphes=True)
    # ME.draw_topks()
    ME.temporal_analysis()
