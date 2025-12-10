import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention
    Optimized for parameter efficiency with adjustable reduction ratio
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Use a higher reduction ratio for larger channels to save parameters
        if channel >= 256:
            reduction = 32
        elif channel >= 128:
            reduction = 16
        else:
            reduction = 8
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # Remove bias to save parameters
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # Remove bias to save parameters
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PyramidPooling(nn.Module):
    """
    Pyramid Pooling Module for multi-scale feature extraction
    Optimized for parameter efficiency with reduced feature channels
    """
    def __init__(self, in_channels, reduction_factor=4):
        super(PyramidPooling, self).__init__()
        # Reduce channels in each pooling branch to save parameters
        reduced_channels = in_channels // reduction_factor
        
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        
        # Use 1x1 convolutions to reduce channels before concatenation
        self.conv1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        
        # Final 1x1 convolution to merge features
        self.final_conv = nn.Conv2d(reduced_channels * 3, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        
        # Apply pooling and channel reduction
        feat1 = self.conv1(self.pool1(x))
        feat2 = self.conv2(self.pool2(x))
        feat3 = self.conv3(self.pool3(x))
        
        # Upsample to original size
        feat1 = F.interpolate(feat1, size=(h, w), mode='bilinear', align_corners=False)
        feat2 = F.interpolate(feat2, size=(h, w), mode='bilinear', align_corners=False)
        feat3 = F.interpolate(feat3, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate and merge
        out = self.final_conv(torch.cat([feat1, feat2, feat3], dim=1))
        out = self.bn(out)
        out = self.relu(out)
        
        # Add residual connection to preserve information
        return out + x

class DoubleConv(nn.Module):
    """
    Double Convolution block with parameter-efficient design
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling block with SE attention
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            SEBlock(out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upsampling block with SE attention
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # Use bilinear upsampling to save parameters
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # When using bilinear upsampling, the number of channels is not halved automatically
            # So we need to account for this in the conv layer
            self.conv = nn.Sequential(
                DoubleConv(in_channels, out_channels),
                SEBlock(out_channels)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                DoubleConv(in_channels, out_channels),
                SEBlock(out_channels)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Adjust dimensions if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SE_PP_UNet(nn.Module):
    """
    U-Net architecture with Squeeze-and-Excitation blocks and Pyramid Pooling Module
    Designed to be lightweight (<2M parameters) and efficient for rock segmentation
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, base_channels=16):
        super(SE_PP_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Use smaller base channel count to reduce parameters
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        self.down3 = Down(base_channels*4, base_channels*8)
        
        # Bottleneck with Pyramid Pooling
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels*8, base_channels*16//factor)
        self.ppm = PyramidPooling(base_channels*16//factor)
        
        # Upsampling path with SE blocks
        self.up1 = Up(base_channels*16, base_channels*8//factor, bilinear)
        self.up2 = Up(base_channels*8, base_channels*4//factor, bilinear)
        self.up3 = Up(base_channels*4, base_channels*2//factor, bilinear)
        self.up4 = Up(base_channels*2, base_channels, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x5 = self.down4(x4)
        x5 = self.ppm(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

    def count_parameters(self):
        """Count the total number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Function to test the model and count parameters
def test_model():
    # Test with different base channel configurations to find optimal size
    base_channels_options = [8, 12, 16, 24, 32]
    results = []
    
    for base_channels in base_channels_options:
        model = SE_PP_UNet(base_channels=base_channels)
        total_params = model.count_parameters()
        
        print(f"Base channels: {base_channels}, Total parameters: {total_params:,}")
        results.append((base_channels, total_params))
        
        # Test with a sample input if under 2M parameters
        if total_params < 2_000_000:
            try:
                x = torch.randn(1, 1, 256, 256)
                y = model(x)
                print(f"  Input shape: {x.shape}")
                print(f"  Output shape: {y.shape}")
                print(f"  Test passed: Yes")
            except Exception as e:
                print(f"  Test failed: {str(e)}")
    
    # Find the largest base_channels that still keeps parameters under 2M
    valid_configs = [(bc, params) for bc, params in results if params < 2_000_000]
    if valid_configs:
        best_config = max(valid_configs, key=lambda x: x[0])
        print(f"\nRecommended configuration: base_channels={best_config[0]} with {best_config[1]:,} parameters")
        return best_config[0], best_config[1] < 2_000_000
    else:
        print("\nNo configuration found under 2M parameters. Further optimization needed.")
        return None, False

if __name__ == "__main__":
    best_base_channels, is_lightweight = test_model()
    if best_base_channels:
        print(f"Model meets lightweight requirement (<2M parameters): {is_lightweight}")
        print(f"Final recommended base_channels: {best_base_channels}")
