from data import parse_data
from data import PascalDataset
from torch.utils.data import DataLoader
from train import train
import torch
from model import SSD300
from commons import *
from detect import detect_single_image, camera
from PIL import Image
from evaluate import test


def main():
    # parse_data("./data/VOC2007/","./data/VOC2012/", "./output/")
    train(resume=True, seed=94743, ignore_first=False, opt_level='O1')
    # image = "/home/numan947/MyHome/AIMLDL/Re:SSD - An Implementation of Single Shot MultiBox Detector/data/VOC2007/JPEGImages/000217.jpg"
    # image = "/home/numan947/Desktop/asdf.jpg"
    # original_image = Image.open(image, mode='r')
    # original_image = original_image.convert('RGB')
    # model = SSD300(21)
    # state_dict = torch.load("./output/checkpoint.pt", map_location=device)
    # model.load_state_dict(state_dict)
    # # test(model, "ALLPT")
    # model = model.to(device)
    # detect_single_image(model, original_image, min_score=0.25, max_overlap=0.4, top_k=200).show()
    # camera(None,None,None,None)

# 2010_001676
# 2010_001687
# 2010_001689
# 2010_001817
# 2010_001841
# 2010_001842
# 2011_001253
# 2011_001254
# 2011_001255
# 2011_001257
if __name__ == "__main__":
    main()