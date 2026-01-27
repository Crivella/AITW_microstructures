import torch
from torch import nn

class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding="same")
        self.relu1a = nn.ReLU()
        self.conv1b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding="same")
        self.relu1b = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding="same")
        self.relu2a = nn.ReLU()
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding="same")
        self.relu2b = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding="same")
        self.relu3a = nn.ReLU()
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding="same")
        self.relu3b = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv4a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding="same")
        self.relu4a = nn.ReLU()
        self.conv4b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding="same")
        self.relu4b = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv5a = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), padding="same")
        self.relu5a = nn.ReLU()
        self.conv5b = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), padding="same")
        self.relu5b = nn.ReLU()

        self.up6 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2,2), stride=(2,2))
        self.conv6a = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3,3), padding="same")
        self.relu6a = nn.ReLU()
        self.conv6b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding="same")
        self.relu6b = nn.ReLU()

        self.up7 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=(2,2))
        self.conv7a = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), padding="same")
        self.relu7a = nn.ReLU()
        self.conv7b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding="same")
        self.relu7b = nn.ReLU()

        self.up8 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=(2,2))
        self.conv8a = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), padding="same")
        self.relu8a = nn.ReLU()
        self.conv8b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding="same")
        self.relu8b = nn.ReLU()

        self.up9 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=(2,2))
        self.conv9a = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), padding="same")
        self.relu9a = nn.ReLU()
        self.conv9b = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3,3), padding="same")

        self.sigmoid10 = nn.Sigmoid()

    def addPadding(self, layer, h1, w1, level):
        h2, w2 = int(h1/2), int(w1/2)
        h3, w3 = int(h2/2), int(w2/2)
        h4, w4 = int(h3/2), int(w3/2)
        h5, w5 = int(h4/2), int(w4/2)

        h = [h1, h2, h3, h4, h5]
        w = [w1, w2, w3, w4, w5]

        th = h[level-1]
        tw = w[level-1]

        lsize = layer.size()
        sh = lsize[2]
        sw = lsize[3]

        zeroPad = nn.ZeroPad2d((tw-sw,0,th-sh,0))
        output = zeroPad(layer)

        return output

    def forward(self, x1, x2, x3):
        lsize = x1.size()
        h1 = lsize[2]
        w1 = lsize[3]

        x = torch.cat((x1,x2,x3), dim=1)

        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        conv1 = self.relu1b(x)
        x = self.pool1(conv1)

        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        conv2 = self.relu2b(x)
        x = self.pool2(conv2)

        x = self.conv3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        conv3 = self.relu3b(x)
        x = self.pool3(conv3)

        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        conv4 = self.relu4b(x)
        x = self.pool4(conv4)

        x = self.conv5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.relu5b(x)

        x = self.up6(x)
        x = self.addPadding(x, h1, w1, level = 4)
        x = torch.cat((x, conv4), 1)
        x = self.conv6a(x)
        x = self.relu6a(x)
        x = self.conv6b(x)
        x = self.relu6b(x)

        x = self.up7(x)
        x = self.addPadding(x, h1, w1, level = 3)
        x = torch.cat((x, conv3), 1)
        x = self.conv7a(x)
        x = self.relu7a(x)
        x = self.conv7b(x)
        x = self.relu7b(x)

        x = self.up8(x)
        x = self.addPadding(x, h1, w1, level = 2)
        x = torch.cat((x, conv2), 1)
        x = self.conv8a(x)
        x = self.relu8a(x)
        x = self.conv8b(x)
        x = self.relu8b(x)

        x = self.up9(x)
        x = self.addPadding(x, h1, w1, level = 1)
        x = torch.cat((x, conv1), 1)
        x = self.conv9a(x)
        x = self.relu9a(x)
        x = self.conv9b(x)

        x = self.sigmoid10(x)

        return x
