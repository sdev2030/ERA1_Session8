import torch
import torch.nn as nn
import torch.nn.functional as F
# dropout_value = 0.025
dropout_value = 0.05

# # Separate 6 channels into 3 groups
# >>> m = nn.GroupNorm(3, 6)
# model = LayerNormCnn(in_shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)

        ) # output_size = 30

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)

        ) # output_size = 28
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6

        
        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 14

        # OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 4

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 2
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 5

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=9)
        ) # output_size = 1

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool1(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class Net_gn(nn.Module):
    def __init__(self):
        super(Net_gn, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)

        ) # output_size = 30

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
#             nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)

        ) # output_size = 28
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 24),
#             nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
#             nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6

        
        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 14

        # OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 24),
#             nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 4

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
#             nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 2
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 5

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=9)
        ) # output_size = 1

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool1(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
class Net_ln(nn.Module):
    def __init__(self):
        super(Net_ln, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 16, affine=False),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)

        ) # output_size = 30

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32, affine=False),
#             nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)

        ) # output_size = 28
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 16, affine=False),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 28

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 16, affine=False),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 24, affine=False),
#             nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32, affine=False),
#             nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 16, affine=False),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6

        
        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 14

        # OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 24, affine=False),
#             nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 4

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 32, affine=False),
#             nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 2
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1, 16, affine=False),
#             nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 5

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=9)
        ) # output_size = 1

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool1(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
# class Net_ln(nn.Module):
#     def __init__(self):
#         super(Net_ln, self).__init__()
#         # Input Block
#         self.convblock1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 4),
# #             nn.LayerNorm([4, 32, 32]),
# #             nn.GroupNorm(4, 16),
# #             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value)

#         ) # output_size = 32

#         # CONVOLUTION BLOCK 1
#         self.convblock2 = nn.Sequential(
#             nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 4),
# #             nn.LayerNorm([4, 32, 32]),
# #             nn.GroupNorm(4, 32),
# #             nn.BatchNorm2d(32),
#             nn.Dropout(dropout_value)

#         ) # output_size = 32
        
#         self.convblock3 = nn.Sequential(
#             nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 1), padding=0, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 4),
# #             nn.LayerNorm([4, 32, 32]),
# #             nn.GroupNorm(4, 16),
# #             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value)
#         ) # output_size = 32

#         # TRANSITION BLOCK 1
#         self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16

#         self.convblock4 = nn.Sequential(
#             nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 8),
# #             nn.LayerNorm([8, 16, 16]),
# #             nn.GroupNorm(4, 16),
# #             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value)
#         ) # output_size = 16

#         # CONVOLUTION BLOCK 2
#         self.convblock5 = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 8),
# #             nn.LayerNorm([8, 16, 16]),
# #             nn.GroupNorm(4, 24),
# #             nn.BatchNorm2d(24),
#             nn.Dropout(dropout_value)
#         ) # output_size = 16
#         self.convblock6 = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 8),
# #             nn.LayerNorm([8, 16, 16]),
# #             nn.GroupNorm(4, 32),
# #             nn.BatchNorm2d(32),
#             nn.Dropout(dropout_value)
#         ) # output_size = 16
#         self.convblock7 = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 8),
# #             nn.LayerNorm([8, 16, 16]),
# #             nn.GroupNorm(4, 16),
# #             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value)
#         ) # output_size = 16

        
#         # TRANSITION BLOCK 2
#         self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8

#         # OUTPUT BLOCK
#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 10),
# #             nn.LayerNorm([10, 8, 8]),
# #             nn.GroupNorm(4, 24),
# #             nn.BatchNorm2d(24),
#             nn.Dropout(dropout_value)
#         ) # output_size = 8

#         self.convblock9 = nn.Sequential(
#             nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 10),
# #             nn.LayerNorm([10, 8, 8]),
# #             nn.GroupNorm(4, 32),
# #             nn.BatchNorm2d(32),
#             nn.Dropout(dropout_value)
#         ) # output_size = 8
#         self.convblock10 = nn.Sequential(
#             nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
#             nn.ReLU(),
#             nn.GroupNorm(1, 10),
# #             nn.LayerNorm([10, 8, 8]),
# #             nn.GroupNorm(4, 16),
# #             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value)
#         ) # output_size = 8

#         self.gap = nn.Sequential(
#             nn.AvgPool2d(kernel_size=8)
#         ) # output_size = 1

#         self.convblock11 = nn.Sequential(
#             nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
#             # nn.BatchNorm2d(10), NEVER
#             # nn.ReLU() NEVER!
#         ) # output_size = 1

#     def forward(self, x):
#         x = self.convblock1(x)
#         x = x + self.convblock2(x)
#         x = x + self.convblock3(x)
#         x = self.pool1(x)
#         x = self.convblock4(x)
#         x = x + self.convblock5(x)
#         x = x + self.convblock6(x)
#         x = x + self.convblock7(x)
#         x = self.pool1(x)
#         x = self.convblock8(x)
#         x = x + self.convblock9(x)
#         x = x + self.convblock10(x)
#         x = self.gap(x)
#         x = self.convblock11(x)
#         x = x.view(-1, 10)
#         return F.log_softmax(x, dim=-1)