import multiprocessing
import os, re
from unittest import result
import numpy as np
import json
import pandas as pd
import torch
from tqdm import tqdm
import sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from itertools import zip_longest
from googleapiclient import discovery


train_posts_dir = ""
test_posts_dir = ""
write_into_train_dir = ""
write_into_test_dir = ""

API_KEY = ""

def request_line(text):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
        )


    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'IDENTITY_ATTACK': {},
                                    'SEVERE_TOXICITY': {},
                                    'INSULT':{},
                                    'SEXUALLY_EXPLICIT':{},
                                    'THREAT': {}}
        }

    response = client.comments().analyze(body=analyze_request).execute()
    return response["attributeScores"] # return a dict


def process_chunk(line):
    line = json.loads(line)
    save_dict = {}
    save_dict["com"] = line["com"]
    save_dict["perspective"] = {}
    try:
        res = request_line(line["com"])
        keys = list(res.keys())
        for k in keys:
            save_dict["perspective"][k] = float(res[k]["summaryScore"]["value"])
    except:
        save_dict["perspective"]=None

    return save_dict

def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


if __name__=="__main__":
    N = 20
    p = multiprocessing.Pool(N)

    with open(test_posts_dir, "r") as file, open(write_into_test_dir, "w") as newfile:
        chunks = file.readlines()
        for chunk in tqdm(grouper(1000, chunks)):
            results = p.map(process_chunk, chunk)
            for res in results:
                try:
                    newfile.write(json.dumps(res))
                    newfile.write("\n")
                except:
                    newfile.write("None")
                    newfile.write("\n")

