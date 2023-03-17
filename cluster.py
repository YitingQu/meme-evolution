import sys, os
import numpy as np
from PIL import Image
from torchvision import transforms
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import pandas as pd
import config as CFG
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import json, random
import matplotlib.pyplot as plt
from utils import *
import ast
import argparse
import networkx as nx
try:
    import cudf
    from cuml import DBSCAN # gpu accelerated DBSCAN
    from cuml import TSNE
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
except:
    pass

np.random.seed = 2022
random.seed = 2022

class ClusterEmbeddings:
    """
    Cluster the embedding with DBSCAN
    embedding modality: image_only, text_only, fused (image+text)
    
    Parameters
    ----------
    embeddings_dir: 4chan image embeddings, text embedding, obtained from inference.py
    eps: DBSCAN parameter, eps distance
    nim_samples: DBSCAN parameter, mininum samples to define a core sample

    Functions
    ----------
    cluster_DBSCAN: perform DBSCAN clustering on embeddings of different modalities

    Outputs
    ----------
    predicted cluter labels, (same order as the meme embeddings), saved in .npy 
    """
    def __init__(self, CFG):
        super(ClusterEmbeddings).__init__()
        self.dtype = np.float32
        self.data = np.load(CFG.embeddings_dir)

    def PCA_reduce(self, embeddings, new_dim=128):
        reduced_embeddings = PCA(n_components=new_dim).fit_transform(embeddings)
        return reduced_embeddings

    def cluster_DBSCAN(self, modality, eps=None, min_samples=None, metric="euclidean"):

        if modality == "image_only":
            embeddings = np.asarray(self.data["image"], dtype=self.dtype)
        elif modality == "text_only":
            embeddings = np.asarray(self.data["text"], dtype=self.dtype)
        elif modality == "fused":
            embeddings = np.asarray(self.data["image"]+self.data["text"], dtype=self.dtype)
        else:
            raise Exception("wrong modality!")
 
        X = self.PCA_reduce(embeddings) # reduce the feature dimension if too much data
        # X = embeddings

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, output_type="numpy", metric=metric).fit(X, out_dtype="int64")
        cluster_labels = dbscan.labels_
        core_samples = dbscan.core_sample_indices_
  
        noise_indices = np.where(cluster_labels==-1)[0]
        noise_mask = np.isin(np.arange(X.shape[0]), noise_indices)
        print(f"there are {len(noise_indices)} noise data")
        print(f"there are {len(np.unique(cluster_labels))} clusters")
        np.save(f"{CFG.save_dir}/dbscan_{modality}", cluster_labels)


class ClusterAnalysis:
    """We conduct analysis of the meme clusters in this class.

    Parameters
    ----------
    save_dir: directory to save results
    embeddings_dir: 4chan embeddings
    cluster_labels_dir: cluster labels, obtained from class ClusterEmbeddings
    data_file: 4chan data, containing textal post and image locations
    perspective_dir: text toxicity file predicted with Google Perspective API
    rewire_dir: text toxicity file predicted with Rewire API
    
    We wrap the above setups to config.

    Functions
    ----------
    extract_keywords: extract_keywords using three different extraction methods
    automatic_annotation: annotate each cluster with key-phrases
    annotation_visualize: tsne visualization of the top-100 clusters
    build_cluster_graph: build a graph with cluster centroids as node

    Outputs
    ----------
    cluster_annotation_hate_scores.xlsx
    tsne.pdf
    graph.gexf

    """

    def __init__(self, CFG):
        super(ClusterAnalysis).__init__()
        self.base_dir  = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = CFG.save_dir

        embeddings = np.load(CFG.embeddings_dir, allow_pickle=True)
        self.image_embeddings, self.text_embeddings = embeddings["image"], embeddings["text"]
        self.cluster_labels =  np.load(CFG.cluster_labels_dir, allow_pickle=True)
        self.counter = Counter(self.cluster_labels).most_common() # [(real_cluster_id, num), ...]
        assert len(self.cluster_labels) == len(self.image_embeddings) == len(self.text_embeddings)

        with open(CFG.data_file, "r") as file:
            self.posts = file.readlines() # line_idx
            file.close()
        with open(CFG.perspective_dir, "r") as file1:
            self.perspective = file1.readlines()
            file1.close()
        with open(CFG.rewire_dir, "r") as file2:
            self.rewire = self.build_dict(file2.readlines()) # a dict with line_idx as the key
            file2.close()

    def build_dict(self, rewire_file):
        rewire_dict = {}
        for line in rewire_file:
            line = json.loads(line)
            rewire_dict[line["line_idx"]] = line
        return rewire_dict

    def cluster_labels_(self):
        return np.unique(self.cluster_labels)

    def read_by_cluster_id(self, cluster_id):
        real_id = self.counter[cluster_id][0]
        indices = np.where(self.cluster_labels==real_id)[0]
        return indices # indices of (images and text) in one cluster

    def retrieve_text_by_clusterID(self, cluster_id, if_topk=False):
        indices = self.read_by_cluster_id(cluster_id)
        if len(indices) >=300 and if_topk:
            indices = np.random.choice(indices, 300, replace=False)
        text = [json.loads(line)["com"] for line in np.array(self.posts)[indices]]
        
        return indices, text

    def retieve_text_by_similarity(self, cluster_id, topk=300):
        indices = self.read_by_cluster_id(cluster_id)
        centroid = np.mean(self.image_embeddings[indices], axis=0)
        centroid = np.expand_dims(centroid, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        norm_text_embeddings = self.text_embeddings / np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        similarities = centroid @ norm_text_embeddings.T # [1, len(embeddings)]
        topk_indices = np.argsort(np.squeeze(similarities))[::-1][:topk] # indices of topk similarites
        text = [json.loads(line)["com"] for line in np.array(self.posts)[topk_indices]]
        return topk_indices, text

    def extract_keywords(self, topk=100, random=False, mode="by_index"):
        # select the cluster with more than 30 samples, and also filter out super large cluster, normally the noise cluster
        select_clusters_ids = np.array([(i, j) for (i, j) in self.counter if j >=30 and j<= 100000])
        if random:
            _ids = np.random.choice(select_clusters_ids.shape[0], topk, replace=False)
            _clusters_ids = select_clusters_ids[_ids]
        else:
            _clusters_ids = select_clusters_ids[:topk]

        res = []
        for real_id, num in tqdm(_clusters_ids):
            indices = np.where(self.cluster_labels==real_id)[0]
            nominal_id = self.counter.index((real_id, num))
            if mode == "by_index":
                _, texts = self.retrieve_text_by_clusterID(cluster_id=nominal_id, if_topk=False)
            elif mode == "by_similarity":
                _, texts = self.retieve_text_by_similarity(cluster_id=nominal_id, topk=300)
            else:
                raise Exception("wrong mode!")
            text = [" ".join(clean_text([com])) for com in texts]

            keyphrases_0 = extract_keywords_keybert(text=" ".join(text), extract_rule="vectorize", top_n=3)
            keyphrases_1 = extract_keywords_keybert(text=" ".join(text), extract_rule="ngram", top_n=3)
            keyphrases_2 = extract_keywords_textrank(text=" ".join(text), top_n=3)
            
            res.append([nominal_id, keyphrases_0, keyphrases_1, keyphrases_2, len(indices)])
            print(res)

        save_df = pd.DataFrame(res, columns=["nominal_id", "vector", "ngram", "textrank", "len"])
        save_df.to_excel(f"{self.save_dir}/cluster_annotation_compare_{mode}.xlsx")


    def automatic_annotation(self, mode="by_similarity"):
        select_clusters_ids = np.array([(i, j) for (i, j) in self.counter if j >=30 and j<= 100000])

        res = []
        for real_id, num in tqdm(select_clusters_ids):
            indices = np.where(self.cluster_labels==real_id)[0]
            nominal_id = self.counter.index((real_id, num))
            if mode == "by_index":
                _, texts = self.retrieve_text_by_clusterID(cluster_id=nominal_id, if_topk=False)
            elif mode == "by_similarity":
                _, texts = self.retieve_text_by_similarity(cluster_id=nominal_id, topk=300)
            else:
                raise Exception("wrong mode!")
            text = [" ".join(clean_text([com])) for com in texts]

            hate_score = 0
            for i in indices:
                identity_attack, rewire_hate = 0, 0
                line = json.loads(self.posts[i])
                perspective_line = json.loads(self.perspective[i])
                try:
                    attack = float(perspective_line["perspective"]["IDENTITY_ATTACK"])
                    if attack >= 0.7:
                        identity_attack += 1
                    rewire_line = self.rewire[str(i)]
                    rewire_label, confidence_score = rewire_line["label"], float(rewire_line["confidence_score"])
                    if rewire_label == "hateful":
                        rewire_hate += 1
                    if (identity_attack + rewire_hate >=1):
                        hate_score += 1
                except:
                    continue
                assert line["com"] == perspective_line["com"] == rewire_line["text"]

            mean_hate_score = hate_score/len(indices)
            keyphrases = extract_keywords_keybert(text=" ".join(text), extract_rule="ngram", top_n=3)
            res.append([nominal_id, keyphrases, len(indices), mean_hate_score])
        
        save_df = pd.DataFrame(res, columns=["nominal_id",  "ngram",  "num_samples", "hate_score"])
        save_df.to_excel(f"{self.save_dir}/cluster_annotation_hate_scores.xlsx")

        self.annotation_file = f"{self.save_dir}/cluster_annotation_hate_scores.xlsx"
    
    
    def process_keyphrase(self, lst):
        noise_lst = ["haha", "nnnn", "aaa", "eee", "ttt", "ooo", "ddd", "ggg"]
        phrase = None
        new_lst = []
        for _phrase in lst:
            if all(noise not in _phrase for noise in noise_lst):
                new_lst.append(_phrase)
        try: 
            for _phrase in new_lst:
                if len(_phrase.split())==3:
                    phrase = _phrase
                    break
            else:
                phrase = new_lst[0]
        except:
            phrase = lst[1]
        tokens = phrase.split()
        return "-".join(clean_text(tokens))
    
    def embedding_annotation_mapping(self, modality="fused", keyphrase_method="ngram", return_toxicity=False):
        annotations_df = pd.read_excel(self.annotation_file)
        cluser_ids = annotations_df["nominal_id"].values
        unique_embeddings, cluster_keyphrases, hate, size = [], [], [], []
        for cluster_id in cluser_ids:
            indices = self.read_by_cluster_id(cluster_id)
            if modality == "image":
                cluster_embeddings = self.image_embeddings[indices]
            elif modality == "fused":
                cluster_embeddings = self.image_embeddings[indices] + self.text_embeddings[indices]
            unique_embeddings.append(np.mean(cluster_embeddings, axis=0))
            row_in_df = annotations_df.loc[annotations_df["nominal_id"]==cluster_id]
            _annotation = row_in_df[keyphrase_method].values[0]
            _annotation = ast.literal_eval(_annotation)
            keyphrase = self.process_keyphrase(_annotation)
            cluster_keyphrases.append(keyphrase)
            # identity_attack = row_in_df["identity_attack"].values[0]
            # rewire_hate = row_in_df["rewire_hate"].values[0]          
            hate.append(float(row_in_df["hate_score"].values[0]))
            size.append(int(row_in_df["num_samples"].values[0]))
        unique_embeddings = np.array(unique_embeddings)
        cluster_keyphrases = np.array(cluster_keyphrases)
        hate, size = np.array(hate), np.array(size)

        if return_toxicity:
            return unique_embeddings, cluster_keyphrases, hate, size
        else:
            return unique_embeddings, cluster_keyphrases

    def annotation_visualize(self, modality="fused", keyphrase_method="ngram", perplexity=10, topk=100):
        unique_embeddings, cluster_keyphrases = self.embedding_annotation_mapping(modality, keyphrase_method)
        unique_embeddings, cluster_keyphrases = unique_embeddings[:topk], cluster_keyphrases[:topk]

        n_components = 2
        tsne = TSNE(n_components=n_components, 
                            perplexity=perplexity,
                            n_iter=10000,
                            metric="cosine",
                            random_state=2022)
        pos = tsne.fit_transform(np.array(unique_embeddings))
        labels = cluster_keyphrases

        fig = plt.figure( figsize=(20, 20), frameon=False)
        K = 15
        FONTSIZE = 20
        scatter_size = np.arange(1, K*len(labels), K)[::-1]

        for i, la in enumerate(labels):
            plt.scatter(pos[i,0], pos[i,1], s=scatter_size[i], alpha=0.4, edgecolor="white")
            if la in ["trudeau-prime", "nazi-pepes"]:
                plt.annotate(la, (pos[i,0], pos[i,1]), c="red", fontsize=FONTSIZE, bbox=dict(facecolor="none", edgecolor="red"))
            else:
                plt.annotate(la, (pos[i,0], pos[i,1]),fontsize=FONTSIZE)


        plt.axis("off")
        # plt.subplots_adjust(right=1.2)
        plt.show()
        fig.savefig(f"{CFG.save_dir}/tsne_{modality}.pdf", bbox_inches="tight", dpi=2000)

        return pos, cluster_keyphrases
        
    def build_cluster_graph(self, modality="fused", keyphrase_method="ngram", extract_hate=False, save_graph=True):
        unique_embeddings, cluster_keyphrases, toxicity, size = self.embedding_annotation_mapping(
                                                                                    modality, 
                                                                                    keyphrase_method, 
                                                                                    return_toxicity=True)
        if extract_hate:
            indices = np.where(toxicity>=0.3)
            unique_embeddings, cluster_keyphrases = unique_embeddings[indices], cluster_keyphrases[indices]
            toxicity, size = toxicity[indices], size[indices]

        unique_embeddings /= np.linalg.norm(unique_embeddings, axis=1, keepdims=True)
        cosine_matrix = unique_embeddings @ unique_embeddings.T
        threds = []
        for row in cosine_matrix:
            threds.append(np.percentile(row, 98.5))
        threds = np.array(threds)
  
        G = nx.Graph()
        for idx in range(cosine_matrix.shape[0]):
            G.add_node(idx, label=cluster_keyphrases[idx], toxicity=toxicity[idx], size=size[idx])
        
        for idx in range(cosine_matrix.shape[0]):
            indices = np.where(cosine_matrix[idx]>=threds[idx])[0]
            for ndx in indices:
                if (idx, ndx) not in G.edges():
                    G.add_edge(idx, ndx, weight=cosine_matrix[idx][ndx])
        if save_graph:
            nx.write_gexf(G, f"{CFG.save_dir}/graph.gexf")
        return G


def main(CFG):

    CE = ClusterEmbeddings(CFG)

    # adjust the eps based on the modality
    # compare the number of clusters and the noise level to the previous studies
    CE.cluster_DBSCAN(modality="image_only", eps=CFG.eps, min_samples=CFG.min_samples)
    CE.cluster_DBSCAN(modality="text_only", eps=CFG.eps, min_samples=CFG.min_samples)
    CE.cluster_DBSCAN(modality="fused", eps=CFG.eps, min_samples=CFG.min_samples)

    CFG.cluster_labels_dir = f"{CFG.save_dir}/dbscan_fused.npy"

    CA = ClusterAnalysis(CFG)

    # compare three types of keyphrase extraction
    # CA.extract_keywords(mode="by_similarity")

    # annotate each cluster with keyphrases extracted from the text
    CA.automatic_annotation()

    # tsne visualization of the top-100 clusters
    # each cluster is represented with its centroid (image/fused) embedding
    CA.annotation_visualize(modality="image")
    CA.annotation_visualize(modality="fused")

    # Build a graph with cluster centroids as node
    # the output .gexf can be processed with gephi
    CA.build_cluster_graph()

   

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        type=str,
        nargs="?",
        default="data/4chan.txt",
        help="4chan txt file"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        nargs="?",
        default="data/embeddings.npz",
        help="image/text embeddings of 4chan data"
    )
    parser.add_argument(
        "--perspective_dir",
        type=str,
        nargs="?",
        default="data/perspective.txt",
        help="perspective toxicity scores of each post/comment"
    )
    parser.add_argument(
        "--rewire_dir",
        type=str,
        nargs="?",
        default="data/rewire.txt",
        help="rewire toxicity scores of each post/comment"
    )
    parser.add_argument(
        "--eps",
        type=float,
        nargs="?",
        default=3,
        help="DBSCAN parameter, eps distance"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        nargs="?",
        default=5,
        help="DBSCAN parameter, mininum samples to define a core sample"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        nargs="?",
        default="result/multimodal_clusters",
        help="save directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="?",
        default="cuda",
        help="device"
    )

    CFG = parser.parse_args()
    Path(CFG.save_dir).mkdir(exist_ok=True, parents=True)
    main(CFG)
