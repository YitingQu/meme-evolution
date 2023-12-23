import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import clip
import matplotlib
import matplotlib.pyplot as plt
import json
from collections import Counter
from PIL import Image
from tqdm import tqdm
from sklearn import cluster
import networkx as nx
from utils import *
import argparse
try:
    import cudf
    from cuml import DBSCAN # gpu accelerated DBSCAN
    from cuml import TSNE
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
except:
    pass


NUM_ROWS_EACH_PAGE = 20

"""
Two Case Studies

1. Happy Merchant
    img_idx=0,
    variant_lower=0.85
    variant_upper=0.91
    influencer_lower=0.91
    final_thred=0.94
    
2. Pepe the frog
    img_idx=1
    lower_bound=0.93
    higher_bound=0.95
    influencer_lower=0.89
    final_thred=0.96

"""


class VisualRegularity:
    """
    A class identifying hateful meme variants given a hateful meme, e.g., Happy Merchant, by extracting visual semantic regularity.

    Visual semantic regularity with image embeddings only: 
        
        *** original meme + influencer -> variant ***

    Parameters:
    ----------
    topk: how many influencers to identify for a pair of original meme and a variant
    n_clusters: number of clusters of variants (if group variants for a quick check)
    lower_bound: lower bound of sim(original meme, variant), a variant should have things in common with the original meme
    higher_bound: higher bound of sim(original meme, variant), a variant shouldn't be identical to the original meme
    influencer_lower: lower bound of sim(influencer, variant), a variant should also have things in common with the influencer
    final_thred: lower bound of sim(original meme+influencer, variant), to make sure the variant is close to the embedding summation
    image_embeddings: all image embeddings
    image_dict: a file of all images locations/urls
    image_root: root directory of saving images
    save_dir: directory to save results

    Functions
    ----------
    retrieve_variants: retrieve variants given a hateful meme
    retrieve_influencers: retrieve influencers given a hateful meme and its variants 
    save_visualize_variants_influencers: save variants and influencers, visualize pairs
    build_graph: build a graph of variants and influencers, containing top-20 communities

    Outputs
    ----------
    variant ids, influencer ids in .npz
    a graph with memes as nodes and pairing relation as edges   
    visualization of variant-influencer pairs in top-20 communities

    """

    def __init__(self, 
                topk=3, 
                n_clusters=100, 
                lower_bound=0.85, 
                higher_bound=0.91, 
                influencer_lower=0.91, 
                final_thred=0.94, 
                image_embeddings=None, 
                image_dict=None,
                image_root=None,
                save_dir=None):

        self.topk = topk
        self.n_clusters = n_clusters
        self.lower_bound = lower_bound 
        self.higher_bound = higher_bound
        self.influencer_lower = influencer_lower
        self.final_thred = final_thred
        self.image_root = image_root
        self.base_dir  = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.base_dir, save_dir)

        self.image_embeddings = torch.from_numpy(np.load(image_embeddings)).type(torch.float64)
        self.image_norm = self.image_embeddings / torch.norm(self.image_embeddings, dim=-1, keepdim=True)

        with open(image_dict, "r") as file:
            self.image_dictionary = file.readlines()
            file.close()

    def retrieve_variants(self, img_idx, group_variants_into_cluster=False):
        """
        Search for all the (potential) variants given an hateful meme.

        img_idx: hateful meme index, i.e., Happy Merchant.
        group_variants_into_cluster: group the similar variants into one cluster, used for testing
        """

        with torch.no_grad():
            center_img = self.image_norm[img_idx].to(device)
            similarities = torch.matmul(center_img, self.image_norm.T.to(device))
            variant_indices = ((similarities>=self.lower_bound)&(similarities<=self.higher_bound)).nonzero().type(torch.int64) # [num_img, 1]
            variant_indices = variant_indices.squeeze().cpu()

            if group_variants_into_cluster:
                if variant_indices.shape[0] > self.n_clusters:
                    km = cluster.KMeans(n_clusters=self.n_clusters, random_state=0)
                    labels = km.fit_predict(self.image_embeddings[variant_indices])
                    variants = []
                    for la in range(self.n_clusters):
                        _indices = np.where(labels==la)[0]
                        select_idx = np.random.choice(_indices, 1)
                        variants.append(variant_indices[select_idx[0]])
                    variant_indices = torch.stack(variants)
        
        print(f"Found {variant_indices.shape[0]} potential variants!")
        return variant_indices

    def retrieve_influencers(self, img_idx, variant_indices):
   
        center_img = self.image_norm[img_idx]
        candidate_embeddings = self.image_norm[img_idx].repeat(self.image_norm.shape[0], 1).type(torch.float64)
        candidate_embeddings = 0.5*candidate_embeddings + 0.5*self.image_norm
        candidate_embeddings /= torch.norm(candidate_embeddings, dim=-1, keepdim=True)

        self.image_norm = self.image_norm.to(device)
        center_img = center_img.to(device)
        candidate_embeddings = candidate_embeddings.to(device)

        new_variant_indices = []
        influencer_indices = []
        for v_id in tqdm(variant_indices):
        
            variant_img =  self.image_norm[v_id]
            
            sim_variant_candidate = torch.matmul(variant_img, self.image_norm.T)
            # sim_variant_center = torch.matmul(center_img, variant_img.T).repeat(self.topk).cpu().numpy()

            # the influencer image should't be too similar (sim<xx) as the meme variant
            improper_candidate_indices = torch.where((sim_variant_candidate > self.influencer_lower))[0]
            mask = torch.isin(torch.arange(self.image_norm.shape[0]).to(device), improper_candidate_indices)
        
            similarities = torch.matmul(variant_img, candidate_embeddings.T).cpu()
            similarities.masked_fill_(mask.cpu(), 0)
            _topk_similarities, _topk_indices = torch.topk(similarities, self.topk, dim=-1) # i.e., [3], [3]
            _topk_similarities, _topk_indices = _topk_similarities.cpu().numpy(), _topk_indices.cpu().numpy()

            if np.max(_topk_similarities) > self.final_thred:
                influencer_indices.append(_topk_indices)
                new_variant_indices.append(v_id.numpy())
            else:
                continue
        new_variant_indices, influencer_indices = np.array(new_variant_indices), np.array(influencer_indices)
        return new_variant_indices, influencer_indices # influencer ids

    def save_visualize_variants_influencers(self, img_idx, group_variants_into_cluster, show_graphes=True):
        _variant_indices = self.retrieve_variants(img_idx, group_variants_into_cluster)
        variant_indices, influencer_indices = self.retrieve_influencers(img_idx=img_idx,
                                                variant_indices=_variant_indices)

        np.savez(f"{self.save_dir}/variant_influencers.npz", variants=variant_indices, influencers=influencer_indices)

        def _draw_figure_by_page(page, img_idx, variant_ids, influencer_ids):
            plt.figure(figsize=(100, 100))
            for row, (v_id, inf_id) in enumerate(zip(variant_ids, influencer_ids)):
                image_in_row = np.hstack([img_idx, v_id, inf_id])
                for col, idx in enumerate(image_in_row):
                    plt.subplot(20, 2+self.topk, (row * len(image_in_row) + col)+1)
                    _, image = self.visualize_meme(int(idx))
                    plt.imshow(image)
                    plt.xticks([])
                    plt.yticks([])
            plt.savefig(f"{self.save_dir}/summation_images_{page}.pdf")
            plt.close()

        if show_graphes:
            num_pages = variant_indices.shape[0] // NUM_ROWS_EACH_PAGE + 1
            for i in range(num_pages):
                start = i*NUM_ROWS_EACH_PAGE
                end = min((i+1)*NUM_ROWS_EACH_PAGE, variant_indices.shape[0])
                variant_ids, influencer_ids = variant_indices[start:end], influencer_indices[start:end]
                _draw_figure_by_page(i,img_idx, variant_ids, influencer_ids)
        else:
            pass

    def visualize_meme(self, idx):
        line = json.loads(self.image_dictionary[idx])
        image_dir = os.path.join(self.image_root, line["img_loc"].lstrip("/"))
        image = Image.open(image_dir).convert("RGB")
        return line, image
    
    def idx2phash(self, idx):
        return json.loads(self.image_dictionary[idx])["phash"]
    
    def test_phash(self, idx_0, idx_1):
        return self.idx2phash(idx_0) == self.idx2phash(idx_1)

   
    def build_graph(self, img_idx, topk_subgraph=False, community_topk=20):

        import community
        np.random.seed(2022)
        res = np.load(f"{self.save_dir}/variant_influencers.npz")
        variants, influencers = res["variants"], res["influencers"]
        influencers = influencers[:, 0] # top-1 influencer

        G = nx.Graph()
        for i, (v_id, inf_id) in enumerate(zip(variants, influencers)):
            triple_pair = [img_idx, v_id, inf_id]
            for id in triple_pair:
                    G.add_node(id)
                    
            G.add_edge(triple_pair[0], triple_pair[1])
            G.add_edge(triple_pair[1], triple_pair[2])

        communities = community.best_partition(G)   # {node_id: community_id}
        community_counter = Counter(communities.values()).most_common() # [(real_community_id, num)]

        for node in G.nodes:
            real_community_id = communities[node]
            community_id = [i for (i, (id, _)) in enumerate(community_counter) if id == real_community_id]
            G.nodes[node]["community_id"] = community_id[0]
            
        variants_influencers_ids = []
        visualized_phashes = []
        for community_id in range(len(community_counter)):
            if len(variants_influencers_ids) == community_topk:
                break

            influencer_in_community = [node for node in influencers if G.nodes[node]["community_id"]==community_id]
            max_degree = 0
            influencer = 0
            for n in influencer_in_community:
                if not (n == img_idx): # center node
                    if G.degree[n] > max_degree:
                        max_degree = G.degree[n]
                        influencer = n
                    else:
                        continue

            variants_candidate_indices = variants[np.where(influencers==influencer)[0]]

            for _idx in variants_candidate_indices:
                if self.idx2phash(_idx) not in visualized_phashes:
                    visualized_phashes.append(self.idx2phash(_idx))
                    variants_influencers_ids.append([_idx, influencer])
                    break
            else:
                _idx = variants_candidate_indices[0]
                visualized_phashes.append(self.idx2phash(_idx))
                variants_influencers_ids.append([_idx, influencer])

        # visualize the images for topk communities
        for i, pair in enumerate(variants_influencers_ids):
            self.visualize_pair(i, pair[0], pair[1])

        if topk_subgraph:
            new_G = nx.Graph()
            orbit_node = img_idx
            new_G.add_node(orbit_node, community_id=0)
            for i in range(community_topk):
                _node = [(id, n) for (id, n) in enumerate(variants) if (G.nodes[n]["community_id"] == i)&(id != orbit_node)][0]
                new_G.add_node(_node[1], community_id=i)
                new_G.add_node(influencers[_node[0]], community_id=i)
                new_G.add_edge(orbit_node, _node[1])
                new_G.add_edge(_node[1],influencers[_node[0]])

            G = new_G
            print("Built a graph with meme variants and influencers.")
            print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

        nx.write_gexf(G, f"{CFG.save_dir}/community/graph.gexf")
        return G
    
    def visualize_pair(self,community_id, idx_0, idx_1):

        _, img_0 = self.visualize_meme(idx=idx_0)
        _, img_1 = self.visualize_meme(idx=idx_1)

        fig, axes = plt.subplots(1, 2, figsize=(20, 20))

        for ax, img in zip(axes, [img_0, img_1]):
            ax.imshow(img)
            ax.axis("off")

            if img == img_0:
                ax.set_title("Variant", fontsize=100)
            else:
                ax.set_title("Influencer", fontsize=100)
        
        plt.subplots_adjust()
        tgt_path = f"{self.save_dir}/community/community_{community_id}"
        Path(tgt_path).mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{tgt_path}/figure.png", bbox_inches="tight")
        plt.close(fig)
   
        
       
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--meme", 
        choices=["HappyMerchant", "PepeTheFrog"], 
        help="meme name, choose from HappyMerchant and PepeTheFrog", 
        default=None
    )
    parser.add_argument(
        "--lower",
        type=float, 
        help="lower bound of variants", 
        default=0.85
    )
    parser.add_argument(
        "--higher",
        type=float, 
        help="higher bound of variants", 
        default=0.91
    )
    parser.add_argument(
        "--influencer_lower", 
        type=float, 
        help="similairty threshold", 
        default=0.91
    )
    parser.add_argument(
        "--final_thred", 
        type=float,
        help="similairty threshold", 
        default=0.94
    )
    parser.add_argument(
        "--image_embeddings", 
        type=str,
        help="image embedding file", 
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
        help="input text", 
        default="result/visual_regularity"
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

    memes_index = {"HappyMerchant": 0, "PepeTheFrog": 1}

    VR = VisualRegularity(topk=3,
                          n_clusters=100,
                          lower_bound=CFG.lower,
                          higher_bound=CFG.higher,
                          influencer_lower=CFG.influencer_lower,
                          final_thred=CFG.final_thred,
                          image_embeddings=CFG.image_embeddings,
                          image_dict=CFG.image_dict,
                          image_root=CFG.image_root,
                          save_dir=CFG.save_dir)

    device = CFG.device

    VR.save_visualize_variants_influencers(img_idx=memes_index[CFG.meme], group_variants_into_cluster=False, show_graphes=False)
    VR.build_graph(img_idx=memes_index[CFG.meme])