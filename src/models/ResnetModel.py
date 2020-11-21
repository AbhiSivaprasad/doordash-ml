
import torch
import torchvision.models as models

class Model: 

    def __init__(self, group, num_classes, lr=0.1, momentum=0.9, weight_decay=1e-4, arch="resnet18", pretrained=False, cpu=False, eval=False):
        self.group = group
        self.num_classes = num_classes
 
        # create model
        if pretrained: 
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True, num_classes=1000)
            model.fc = torch.nn.Linear(512, num_classes)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch](num_classes=num_classes)
            print("=> finished creating model '{}'".format(arch))
       
        if cpu:
            print("USING CPU")
            self.model = torch.nn.DataParallel(model)
        else:
            print("USING GPU")
            self.model = torch.nn.DataParallel(model).cuda()

        if eval:
            self.model.eval()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)


    def load_model(self, checkpoint_file, cpu=True):
    
        print("=> loading checkpoint '{}'".format(checkpoint_file))

        if cpu:
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(checkpoint_file)
    
        # optionally resume from a checkpoint
        if "labels" in checkpoint:
            labels = checkpoint["labels"]
            print("LABELS", labels)
            num_classes = len(labels)
    
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.labels = sorted(checkpoint["labels"])
