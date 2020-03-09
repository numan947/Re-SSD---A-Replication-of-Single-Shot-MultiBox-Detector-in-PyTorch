from data import parse_data
from data import PascalDataset
from torch.utils.data import DataLoader
from train import train
import torch
from model import SSD300
from commons import *
from detect import detect
from PIL import Image

def main():
    torch.manual_seed(947)
    # parse_data("./data/VOC2007/","./data/VOC2012/", "./output/")
    # train(resume=False)
    image = "/home/numan947/MyHome/AIMLDL/Re:SSD - An Implementation of Single Shot MultiBox Detector/data/VOC2012/JPEGImages/2007_000323.jpg"
    original_image = Image.open(image, mode='r')
    original_image = original_image.convert('RGB')
    model = SSD300(21)
    state_dict = torch.load("./output/checkpoint.pt", map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    detect(model, original_image, min_score=0.15, max_overlap=0.2, top_k=200).show()

if __name__ == "__main__":
    main()