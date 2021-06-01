
import os, gdown
import torchvision
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms


class Net(torch.nn.Module):
    def __init__(self, base_model, base_out_features, num_classes):
        super(Net,self).__init__()
        self.base_model=base_model
        self.linear1 = torch.nn.Linear(base_out_features, 512)
        self.output = torch.nn.Linear(512,num_classes)

    def forward(self,x):
        x = F.relu(self.base_model(x))
        x = F.relu(self.linear1(x))
        x = self.output(x)
        return x


class FinalModel(object):

    def __init__(self):

        res = torchvision.models.resnet50(pretrained=False)
        self.imageTranform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.model = Net(base_model=res, base_out_features=res.fc.out_features, num_classes=10)

        self.labels = {0: 'beagle',
                 1: 'chihuahua',
                 2: 'doberman',
                 3: 'french_bulldog',
                 4: 'golden_retriever',
                 5: 'malamute',
                 6: 'pug',
                 7: 'saint_bernard',
                 8: 'scottish_deerhound',
                 9: 'tibetan_mastiff'}
        modelurl = "https://drive.google.com/uc?id=1f8xV-v8-0_1AxIs4gzqSz4-VzYq5P4w0"
        outputPath = "./dogbreedsTask.pth"
        if (os.path.isfile(outputPath)):
            self.model.load_state_dict(torch.load("./dogbreedsTask.pth",map_location=torch.device("cpu")))
        else:
            gdown.download(modelurl,outputPath,quiet=False)
            self.model.load_state_dict(torch.load("./dogbreedsTask.pth",map_location=torch.device("cpu")))

    def predict(self,image:Image)->(float,str):
        pred = self.model(self.imageTranform(image).unsqueeze(0)).detach().squeeze().softmax(-1)
        idx = pred.argmax(-1)
        score = pred[idx.item()]
        label = self.labels[idx.item()]
        return score.item(),label