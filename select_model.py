import torch
import torch.nn as nn
from utils.Configuration import CFG
from torchvision import models

#model
def choose_model(model_name):
    if model_name == 'VisionTransformer_Base16':
        #Define Pretrained ViT model
        model = models.vit_b_16(pretrained=True)
        model.heads = nn.Sequential(
            nn.Linear(
            in_features=768,
            out_features=7
            #out_features=9
        ))
        model = model.to(CFG.device)

    elif model_name == 'VisionTransformer_Base32':
        #Define Pretrained ViT model
        model = models.vit_b_32(pretrained=True)
        model.heads = nn.Sequential(
            nn.Linear(
            in_features=768,
            out_features=9
        ))
        model = model.to(CFG.device)

    elif model_name == 'VisionTransformer_Large16':
        #Define Pretrained ViT model
        model = models.vit_l_16(pretrained=True)
        model.heads = nn.Sequential(
            nn.Linear(
            in_features=1024,
            out_features=9
        ))
        model = model.to(CFG.device)

    elif model_name == 'VisionTransformer_Large32':
        #Define Pretrained ViT model
        model = models.vit_l_32(pretrained=True)
        model.heads = nn.Sequential(
            nn.Linear(
            in_features=1024,
            out_features=9
        ))
        model = model.to(CFG.device)

    elif model_name == 'VGG16':
        #Define Pretrained VGG16 model
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(
            in_features=4096,
            out_features=9
        )
        model = model.to(CFG.device)

    elif model_name == 'Inceptionv3':
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(
                        in_features=2048,
                        out_features=9
        )
        model = model.to(CFG.device)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Linear(
                        in_features=1280,out_features=9
        )
        model = model.to(CFG.device)

    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=True)
        model.classifier = nn.Linear(
                        in_features=1536,out_features=9
        )
        model = model.to(CFG.device)

    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=True)
        model.classifier = nn.Linear(
                        in_features=2560,out_features=9
        )
        model = model.to(CFG.device)
    return model