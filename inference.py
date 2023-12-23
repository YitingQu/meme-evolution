import os, sys, json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import argparse


class FourChanDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        super(FourChanDataset).__init__()
        """
        parameters
        ----------
        data_dir: a txt file saving text and image locations

        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        print(self.base_dir)
        self.data = open(os.path.join(self.base_dir, data_file), "r").readlines()
        self.transforms = self._transforms()

    @staticmethod
    def _transforms():
        return Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        line = json.loads(self.data[idx]) # read a line and turn it into a dict
        img_dir = os.path.join(self.base_dir, "data/images", line["img_loc"])
        image = Image.open(img_dir).convert("RGB")
        image = self.transforms(image) # [3, 224, 224]
        text = line["com"] # str
        return image, text # tensor type image; text without tokenization


    def __len__(self):
        return len(self.data)
    
class FourChanImage(torch.utils.data.Dataset):
    def __init__(self, data_file):
        super(FourChanImage).__init__()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        print(self.base_dir)
        self.data = open(os.path.join(self.base_dir, data_file), "r").readlines()
        self.transforms = FourChanDataset._transforms()

    def __getitem__(self, idx):
        line = json.loads(self.data[idx]) # read a line and turn it into a dict
        img_dir = os.path.join(self.base_dir, "data/images", line["img_loc"])
        image = Image.open(img_dir).convert("RGB")
        image = self.transforms(image) # [3, 224, 224]
        return image

    def __len__(self):
        return len(self.data)

def inference(CFG):
    
    device = CFG.device
    Path(CFG.save_dir).mkdir(exist_ok=True, parents=True)


    dataset = FourChanDataset(CFG.data_file)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False)

    # load CLIP model
    model, _ = clip.load(CFG.model_file, device=device)
    model = model.eval()

    image_embeddings_all = []   
    text_embeddings_all = []

    with torch.no_grad():
        for  batch in tqdm(dataloader):
            batch_image, batch_txt = batch # [batch_size, 3, 224, 224]; tuple of str
            images = batch_image.to(device)
            texts = clip.tokenize(batch_txt).to(device)
            image_embeddings = model.encode_image(images)
            text_embeddings = model.encode_text(texts)
            image_embeddings_all.append(image_embeddings.detach().cpu().numpy())
            text_embeddings_all.append(text_embeddings.detach().cpu().numpy())

            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

        image_embeddings_all = np.vstack(image_embeddings_all)
        text_embeddings_all = np.vstack(text_embeddings_all)
        print(image_embeddings_all.shape)

        np.savez(f"{CFG.save_dir}/embeddings.npz", image=image_embeddings_all, text=text_embeddings_all)

def inference_images(CFG):

    device = CFG.device
    Path(CFG.save_dir).mkdir(exist_ok=True, parents=True)


    dataset = FourChanImage(CFG.data_file)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False)

    # load CLIP model
    model, _ = clip.load(CFG.model_file, device=device)
    model = model.eval()

    image_embeddings_all = [] 

    with torch.no_grad():
        for  batch in tqdm(dataloader):
            batch_image = batch # [batch_size, 3, 224, 224]; tuple of str
            images = batch_image.to(device)
            image_embeddings = model.encode_image(images)
            image_embeddings_all.append(image_embeddings.detach().cpu().numpy())
           
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

        image_embeddings_all = np.vstack(image_embeddings_all)
        print(image_embeddings_all.shape)

        np.save(f"{CFG.save_dir}/image_embeddings.npy", image_embeddings_all)


def main(CFG):
    
    if CFG.data_file == "data/4chan.txt":
        inference(CFG)
    elif CFG.data_file == "data/4chan_images_only.txt":
        inference_images(CFG)
    else:
        raise Exception("Wrong data file!")        

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
        "--model_file",
        type=str,
        nargs="?",
        default=None,
        help="CLIP model path, load from the fine-tuned checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        default=512,
        help="batch_size when loading 4chan images"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        nargs="?",
        default=2,
        help="number of workers when loading 4chan images"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        nargs="?",
        default="data",
        help="base directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="?",
        default="cuda",
        help="device"
    )

    CFG = parser.parse_args()
    main(CFG)