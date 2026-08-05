import torch
import timm

class KeypointModel(torch.nn.Module):
    def __init__(self, num_classes=2, num_keypoint=8):
        super().__init__()
        self.backbone = timm.create_model('resnet50', features_only=True, pretrained=True)
        self.dim = 2048
        self.class_linear = torch.nn.Linear(self.dim, num_classes)
        self.kp_linear = torch.nn.Linear(self.dim, num_keypoint * 2)

    def forward(self, x):
        bs = x.size(0)
        x = self.backbone(x)[-1]
        x = torch.nn.functional.avg_pool2d(x, (7, 7)).reshape(bs, self.dim)
        cls = self.class_linear(x)
        kp = self.kp_linear(x).sigmoid()
        return cls, kp.reshape(bs, -1, 2)


if __name__ == "__main__":
    model = KeypointModel()
    model.cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    cls, kp = model(x)
    print(cls.shape, kp.shape)
