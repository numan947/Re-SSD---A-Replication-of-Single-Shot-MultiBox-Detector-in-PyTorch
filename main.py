from data import parse_data
from data import PascalDataset
from torch.utils.data import DataLoader


def main():
    # parse_data("./data/VOC2007/","./data/VOC2012/", "./output/")
    
    dataset = PascalDataset("./output/",split="Train")
    dataloader = DataLoader(dataset, batch_size=50, collate_fn=dataset.collate_fn)
    
    for p in dataloader:
        break
    

if __name__ == "__main__":
    main()