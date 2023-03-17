import numpy as np
import pandas as pd
import os, cv2, gc, itertools
import json
from tqdm import tqdm
from pathlib import Path
from itertools import zip_longest
import multiprocessing 

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
json_file = "pol_062016-112019_labeled.ndjson"
match_file = "manifest_all.txt"

mapping = open(os.path.join(base_dir, match_file), "r").readlines()

mapping_dict = {}
for line in tqdm(mapping):
    line = line.rstrip("\n").split("\t")
    mapping_dict[line[3]] = line[2]


def search_image_loc(mapping_dict, md5Str):
    """
    manifest_all.txt: ['post_id', 'image_url', 'image_dir', 'md5_checksum\n']
    """
    try:
        return mapping_dict[md5Str]
    except:
        # print(f"no such image: {md5Str}")
        return None

def process_chunk(thread): # a thread consisting of multiple posts: txt
    if thread:
        thread = json.loads(thread)
        posts = thread["posts"]
        new_threads = []
        for post in posts:
            res = {}
            if all(x in post.keys() for x in ["md5", "com", "perspectives", "time", "entities"]):
                res["id"] = post["no"]
                res["time"] = post["time"]
                res["com"] = post["com"]
                res["entities"] = post["entities"]
                res["perspective"] = post["perspectives"]
                md5str = post["md5"]
                res["img_loc"] = search_image_loc(mapping_dict, md5str)
                if res["img_loc"]:
                    new_threads.append(res)
        return new_threads # a list of dicts

def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

if __name__ == "__main__":
    
    # create pool
    N = 30
    p = multiprocessing.Pool(N)

    #write a txt file line by line
    with open(f"{base_dir}processed_posts_all.txt", "w") as file:
        with open(os.path.join(base_dir, json_file), "r") as jsonfile:
            
            chunks = jsonfile.readlines()
            for chunk in tqdm(grouper(1000, chunks)):
                results = p.map(process_chunk, chunk) # a list of a list of dicts/ a dict
        
                for chunk_res in results: # [{}]; [{}, {},...]
                    if chunk_res:
                        for res in chunk_res:
                            file.write(json.dumps(res))
                            file.write("\n")