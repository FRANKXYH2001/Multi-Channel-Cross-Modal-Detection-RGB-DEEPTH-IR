from ast import Try
from dataloader.casia_surf_dataset_loader import CASIASURFDataset
from loss.cmfl_loss import CMFL
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
import warnings
import os
warnings.filterwarnings("ignore")
from architectures.MultiStreamDenseNet import MultiStreamDenseNet
from architectures.OneStreamDenseNet import OneStreamDenseNet

class TrainerConfig():
    """ 
    This class configures the training of the network and supports different architectures and experiments.
    """

    def __init__(self, train_args):
        self.args = train_args
        
        architectures = dict({'two_stream': MultiStreamDenseNet, 'one_stream': OneStreamDenseNet}) # supported architectures
        loss_types =  {"bce", "cmfl"}
        allowed_channels =  {"rgb", "depth", "ir"}

        self.DATA_DIR = train_args.dataroot
        self.PROTOCOL_DIR = train_args.protocol_dir
        self.PROTOCOL_NAME = train_args.protocol_name
        self.SELECTED_ARCHITECTURE = train_args.architecture
        self.LOSS_TYPE = train_args.loss
        self.PRETRAINED = train_args.pretrained
        self.BRANCH1_CHANNEL = train_args.branch1_channel
        self.BRANCH2_CHANNEL = train_args.branch2_channel
        self.DO_CROSS_VALIDATION = True

        # Validate the input channels
        if self.BRANCH1_CHANNEL not in allowed_channels:
            raise Exception("Unsupported channel type for branch 1")
        if self.BRANCH2_CHANNEL not in allowed_channels:
            raise Exception("Unsupported channel type for branch 2")

        # Validate other configurations
        if self.SELECTED_ARCHITECTURE not in architectures.keys():
            raise Exception("Invalid architecture selected. Please check your configuration")

        if self.LOSS_TYPE not in loss_types:
            raise Exception("Unsupported loss type")

        self.transform = transform = transforms.Compose([
            transforms.Resize((112, 112))
        ])

        phases = ['train', 'val']
        phase_files = {"train": train_args.train_split, "val": train_args.val_split}

        self.dataset = {}

        for phase in phases:
            self.dataset[phase] = CASIASURFDataset(
                self.DATA_DIR, 
                os.path.join(self.PROTOCOL_DIR, self.PROTOCOL_NAME, phase_files[phase]),
                self.transform,
                custom_function,
                is_train=(phase == 'train')
            )

        # Load the architecture
        self.network = architectures[self.SELECTED_ARCHITECTURE](pretrained=self.PRETRAINED)

        for name,param in self.network.named_parameters():
            param.requires_grad = True

        # loss definitions

        self.criterion_bce= nn.BCELoss()
        self.criterion_cmfl = CMFL(alpha=1, gamma= 3, binary= False, multiplier=2)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()),lr = train_args.lr, weight_decay=train_args.weight_decay)

                                    
    def compute_loss(self, network, img, labels, device):
        """
        Compute the losses, given the network, data and labels and 
        device in which the computation will be performed. 
        """
        if self.SELECTED_ARCHITECTURE == "two_stream":
            if self.LOSS_TYPE == "bce":
                return self.two_stream_loss_bce(network, img, labels, device)
            elif self.LOSS_TYPE == "cmfl":
                return self.two_stream_loss_cmfl(network, img, labels, device)
            else:
                raise Exception("unsupported loss")
        elif self.SELECTED_ARCHITECTURE == "one_stream":
            return self.one_stream_loss(network, img, labels, device)
        else:
            raise Exception("Loss is not defined for the selected architecture")

    def two_stream_loss_cmfl(self, network, img, labels, device):
        """
        Loss for the two stream architecture using CMFL
        """
        beta = 0.5
        img1_tensor = Variable(img[self.BRANCH1_CHANNEL].to(device))
        img2_tensor = Variable(img[self.BRANCH2_CHANNEL].to(device))

        labelsv_binary = Variable(labels['binary_target'].to(device))

        op, op_branch1, op_branch2 = network(img1_tensor, img2_tensor)

        loss_cmfl = self.criterion_cmfl(op_branch1, op_branch2, labelsv_binary.unsqueeze(1).float())
        loss_bce = self.criterion_bce(op, labelsv_binary.unsqueeze(1).float())
        loss = beta * loss_cmfl + (1 - beta) * loss_bce

        return loss

    def two_stream_loss_bce(self, network, img, labels, device):
        """
        Loss for the two stream architecture using BCE
        """
        img1_tensor = Variable(img[self.BRANCH1_CHANNEL].to(device))
        img2_tensor = Variable(img[self.BRANCH2_CHANNEL].to(device))

        labelsv_binary = Variable(labels['binary_target'].to(device))

        op, _, _ = network(img1_tensor, img2_tensor)

        loss_bce = self.criterion_bce(op, labelsv_binary.unsqueeze(1).float())

        return loss_bce

    def one_stream_loss(self, network, img, labels, device):
        """
        Loss for the one stream architecture using BCE
        """
        labelsv_binary = Variable(labels['binary_target'].to(device))
        
        if self.SPECTRA == "rgb":
            img_tensor = Variable(img['rgb'].to(device))
        elif self.SPECTRA == "depth":
            img_tensor = Variable(img['depth'].to(device))
        elif self.SPECTRA == "ir":
            img_tensor = Variable(img['ir'].to(device))
        else:
            raise Exception("Unsupported spectra type")

        op = network(img_tensor)
        loss_bce = self.criterion_bce(op, labelsv_binary.unsqueeze(1).float())
        
        return loss_bce

# label: 1 = Real, 0 = attack
def custom_function(img1_tensor, img1_type, img2_tensor, img2_type, label, img_name):
    img = {}
    img[img1_type] = img1_tensor
    img[img2_type] = img2_tensor
    labels={}
    labels['binary_target']=label
    labels['img_name']=img_name
    
    return img, labels