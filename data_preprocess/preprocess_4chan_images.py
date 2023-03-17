from fileinput import filename
from tkinter.font import names
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os, cv2, gc, itertools
import matplotlib.pyplot as plt
import json
import yaml
import re, sys
import random
from PIL import Image
from pathlib import Path
from itertools import zip_longest
import multiprocessing 
from tqdm import tqdm


np.random.seed = 2022
random.seed = 2022

# subset image dir
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_root = os.path.join(base_dir, "data/images") 
raw_posts = os.path.join(base_dir, "data/4chan.txt")
image_dict_dir = os.path.join(base_dir, "data/4chan_images_only.txt")

import warnings
warnings.simplefilter("error", Image.DecompressionBombWarning)

def search_image_loc(mapping_dict, md5Str):
    """
    manifest_all.txt: ['post_id', 'image_url', 'image_dir', 'md5_checksum\n']
    """
    try:
        return mapping_dict[md5Str]
    except:
        # print(f"no such image: {md5Str}")
        return None
    
def load_data(line):
    post = json.loads(line)
    id = post["id"]
    time = post["time"]
    image_dir = post["img_loc"].lstrip("./4chan_images") # 1467/30/img_name
    # image_dir = os.path.join(image_dir[:4], image_dir[:4], image_dir)
    # assert os.path.exists(image_dir)
    return id, time, image_dir

def process_chunk(img_loc):
    res = None
    correct_img = check_bad_image(img_loc)
    if correct_img:
        phash = compute_pash(img_loc)
        # time = 
        if phash:
            res = {}
            res["img_loc"] = img_loc
            res["phash"] = phash
    return res # none or dict

def check_bad_image(img_loc):
    res = True
    if os.path.exists(os.path.join(image_root,img_loc)):
        try:
            img = Image.open(os.path.join(image_root,img_loc)).convert("RGB")
            img.verify()
        except (IOError, SyntaxError, OSError, Warning, UserWarning):
            res = False
    else:
        res = False
    return res

def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def compute_pash(img_loc):
    import imagehash
    try:
        image = Image.open(os.path.join(image_root,img_loc))
        phash = str(imagehash.phash(image))
    except:
        phash = None
    return phash



if __name__ == "__main__":
    # multiprocessing
    N = 50
    p = multiprocessing.Pool(N)

    # split_eval_from_train()
    with open(image_dict_dir, "w") as writefile:
        chunks = [img_name for root, _, filenames in list(os.walk(image_root, topdown=False)) for img_name in filenames]
        #  chunks = [os.path.join(image_root, img_name) for root, _, filenames in list(os.walk(image_root, topdown=False)) for img_name in filenames]
        for chunk in tqdm(grouper(1000, chunks)):
            results = p.map(process_chunk, chunk)
            for res in results:
                if res:
                    writefile.write(json.dumps(res))
                    writefile.write("\n")
