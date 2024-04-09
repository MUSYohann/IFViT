import torch
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F

class LocalizationNetwork(torch.torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The localization net uses a downsampled version of the image for performance
        self.input_size = (128, 128)
        self.resize = torchvision.transforms.Resize(
            size=self.input_size, antialias=True
        )
        # Spatial transformer localization-network
        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = torch.nn.Sequential(
            torch.nn.Linear(8 * 8 * 64, 64), torch.nn.ReLU(True), torch.nn.Linear(64, 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        resized_x = self.resize(x)
        xs = self.localization(resized_x)
        xs = xs.view(-1, 64 * 8 * 8)
 
        theta_1 = self.fc_loc(xs)
   
        theta = torch.zeros(theta_1.shape).cuda()

        
        theta[:,0] = theta_1[:,0]
        theta[:,1] = theta_1[:,1]
        theta[:,2] = theta_1[:,2]


        theta[:, 0] = torch.tanh(theta_1[:, 0]) * 30 * (torch.pi / 180)  
        max_translation = self.input_size[0] * 0.1 
        theta[:, 1] = torch.tanh(theta_1[:, 1]) * max_translation 
        theta[:, 2] = torch.tanh(theta_1[:, 2]) * max_translation


        cos_theta = torch.cos(theta[:,0]).cuda()
        sin_theta = torch.sin(theta[:,0]).cuda()
        translation_x = theta[:,1]
        translation_y = theta[:,2]

        affine_matrices = torch.zeros(x.shape[0], 2, 3).cuda()
        affine_matrices[:, 0, 0] = cos_theta
        affine_matrices[:, 0, 1] = -sin_theta
        affine_matrices[:, 0, 2] = translation_x / self.input_size[0]
        affine_matrices[:, 1, 0] = sin_theta
        affine_matrices[:, 1, 1] = cos_theta
        affine_matrices[:, 1, 2] = translation_y / self.input_size[1]

        grid = F.affine_grid(affine_matrices, x.size(),align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        aligned_img = x[0,0,:,:].cpu().detach().numpy()*255.0
        aligned_img = aligned_img.astype(np.uint8)
        return x


def main():
    model = LocalizationNetwork()
    print("no syntax errors")


if __name__ == "__main__":
    main()
