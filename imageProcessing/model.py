import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomUNet(nn.Module):
    def __init__(self):
        super(CustomUNet, self).__init__()
        
        # --- ENCODER ---
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1) # Strided conv downsampling

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # --- BRIDGE ---
        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # --- DECODER ---
        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 256 due to concat (128 + 128)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 due to concat (64 + 64)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)   # 64 due to concat (32 + 32)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.up9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)   # 32 due to concat (16 + 16)
        self.conv9_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # Output layer
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # --- ENCODER ---
        c1 = F.relu(self.conv1_1(x))
        c1 = F.relu(self.conv1_2(c1))
        p1 = F.relu(self.pool1(c1))

        c2 = F.relu(self.conv2_1(p1))
        c2 = F.relu(self.conv2_2(c2))
        p2 = F.relu(self.pool2(c2))

        c3 = F.relu(self.conv3_1(p2))
        c3 = F.relu(self.conv3_2(c3))
        p3 = F.relu(self.pool3(c3))

        c4 = F.relu(self.conv4_1(p3))
        c4 = F.relu(self.conv4_2(c4))
        p4 = F.relu(self.pool4(c4))

        # --- BRIDGE ---
        c5 = F.relu(self.conv5_1(p4))
        c5 = F.relu(self.conv5_2(c5))

        # --- DECODER ---
        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1) # Concatenate on channel dimension
        c6 = F.relu(self.conv6_1(u6))
        c6 = F.relu(self.conv6_2(c6))

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = F.relu(self.conv7_1(u7))
        c7 = F.relu(self.conv7_2(c7))

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = F.relu(self.conv8_1(u8))
        c8 = F.relu(self.conv8_2(c8))

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = F.relu(self.conv9_1(u9))
        c9 = F.relu(self.conv9_2(c9))

        outputs = torch.sigmoid(self.output(c9))
        return outputs

def build_custom_unet():
    return CustomUNet()