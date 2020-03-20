from data import parse_data
from data import PascalDataset
from torch.utils.data import DataLoader
from train import train
import torch
from model import SSD
from commons import *
from detect import detect_single_image, camera
from PIL import Image
from evaluate import test


def main():
    # # FOR DATA PARSING
    # parse_data("./data/VOC2007/","./data/VOC2012/", "./output/")
    
    # # FOR TRAINING
    train(resume=True, seed=94743, opt_level='O1', resize_dims=(500,500))
    
    # # FOR EVALUATION AND TESTING
    # model = SSD(21)
    # state_dict = torch.load("./output/checkpoint.pt", map_location=device)
    # model.load_state_dict(state_dict)
    # model = model.to(device)
    
    # # FOR EVALUATION ON THE TESTING SET    
    # test(model, "ALLPT", resize_dims=(500,500))

    # # FOR DETECTION ON A SINGLE IMAGE
    # image = "/home/numan947/MyHome/AIMLDL/Re:SSD - An Implementation of Single Shot MultiBox Detector/data/VOC2007/JPEGImages/005284.jpg"
    # original_image = Image.open(image, mode='r')
    # original_image = original_image.convert('RGB')
    # detect_single_image(model, original_image, min_score=0.4, max_overlap=0.1, top_k=200, resize_dims=(500,500)).show()
    
    # # FOR DEMO USING CAMERA
    # camera(model, min_score=0.3, max_overlap=0.25, top_k=20, resize_dims=(500,500))

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