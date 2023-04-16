import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output#, sr_feature

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet101_fpn():
    return _FPN(classes=100, srmode=True)

class _Residual_Block(nn.Module): 
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x): 
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output 


class SR(nn.Module):
    def __init__(self):
        super(SR, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.residual = self.make_layer(_Residual_Block, 2)
        self.conv_mid = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_output = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out,residual)
        out = self.conv_output(out)     
        
        return out

# ----------- fpn
class _FPN(nn.Module):
    def __init__(self, classes, pretrained=False, srmode=True):
        super(_FPN, self).__init__()
        self.n_classes = classes
        self.model_path ='checkpoint/resnet101/lr_resnet101_epoch100/best.pth'
        self.dout_base_model = 256
        self.pretrained = pretrained
        self.srmode=srmode

        resnet = resnet101()
        
        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()}, strict=False)

        self.Resnet_layer0 = nn.Sequential(resnet.conv1)
        self.Resnet_layer1 = nn.Sequential(resnet.conv2_x)
        self.Resnet_layer2 = nn.Sequential(resnet.conv3_x)
        self.Resnet_layer3 = nn.Sequential(resnet.conv4_x)
        self.Resnet_layer4 = nn.Sequential(resnet.conv5_x)


        # Top layer
        self.Resnet_toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        self.Resnet_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.Resnet_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.Resnet_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.Resnet_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.Resnet_latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.Resnet_latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.layer3d = self._make_3dconv_layer(256,256)
        self.fc3d = nn.Linear(256, self.n_classes)

        self.avgpool3d = nn.AdaptiveAvgPool3d((1, None, None))
        self.fc = nn.Linear(512, self.n_classes)
        self.sr = SR()
        

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y      # remove path y


    def _make_3dconv_layer(self, in_c, out_c):
        conv3d_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(out_c, eps=0.001),
            nn.LeakyReLU()
        )
        return conv3d_layer


    def sequential_feature(self, x):
        x = self.layer3d(x)
        x = self.avgpool3d(x)
        x = x.squeeze(2)

        return x


    def forward(self, im_data):

        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.Resnet_layer0(im_data)
        c2 = self.Resnet_layer1(c1)
        c3 = self.Resnet_layer2(c2)
        c4 = self.Resnet_layer3(c3)
        c5 = self.Resnet_layer4(c4)


        #Top-down
        p5 = self.Resnet_toplayer(c5)
        p4 = self._upsample_add(p5, self.Resnet_latlayer1(c4))
        p4 = self.Resnet_smooth1(p4)

        p6 = self.maxpool2d(p5)
        org = p4

        _, _, H, W = p4.size()
        up_p5 = F.upsample(p5, size=(H, W), mode='bilinear')
        up_p6 = F.upsample(p6, size=(H, W), mode='bilinear')


        p4 = p4.unsqueeze(2)
        up_p5 = up_p5.unsqueeze(2)
        up_p6 = up_p6.unsqueeze(2)

        # sequence feature
        general_view = torch.cat((p4, up_p5, up_p6), dim=2)
        fpn_sequential_feature = self.sequential_feature(general_view)

        lr_feature = torch.cat([org, fpn_sequential_feature], dim=1 )

        if self.srmode:
            lr_feature = self.sr(lr_feature)

        fpn_org_sequential_feature = self.avgpool2d(lr_feature)
        fpn_org_sequential_feature=torch.flatten(fpn_org_sequential_feature, 1)

        #sequence feature concat org feature
        fpn_org_sequential_feature = self.fc(fpn_org_sequential_feature)
        
        return fpn_org_sequential_feature, lr_feature



