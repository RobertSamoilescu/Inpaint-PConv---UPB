import numpy as np
import PIL.Image as pil
import os
import matplotlib.pyplot as plt

# monodepth
from .monodepth.inverse_warp import *
from .monodepth.depth_decoder import *
from .monodepth.layers import *
from .monodepth.pose_cnn import *
from .monodepth.pose_decoder import *
from .monodepth.resnet_encoder import *

# inpainting
from .inpaint.net import *
from .inpaint.io import *
from .inpaint.image import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Monodepth(object):
    def __init__(self, model_name: str, root_dir: str="."):
        self.model_name = model_name
        self.root_dir = root_dir
        self.intrinsic = np.array([
            [0.61, 0, 0.5],   # width
            [0, 1.22, 0.5],   # height
            [0, 0, 1]],
        dtype=np.float32)
        self.CAM_HEIGHT = 1.5
        
        encoder_path = os.path.join(root_dir, "models", model_name, "encoder.pth")
        depth_decoder_path = os.path.join(root_dir, "models", model_name, "depth.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = ResnetEncoder(18, False)
        self.encoder = self.encoder.to(device)
        
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.depth_decoder = self.depth_decoder.to(device)
        
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval();
        
        
    def forward(self, img: torch.tensor):
        """
        @param img: input image (RGB), [B, 3, H, W]
        :returns depth map[B, 1, H, W]
        """
        # normalize
        if img.max() > 1:
            img = img / 255.
        
        img = img.to(device)
        
        with torch.no_grad():
            features = self.encoder(img)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)].cpu()
        return disp
    
    def get_depth(self, disp: torch.tensor):
        """
        @param disp: disparity map, [B, 1, H, W]
        :returns depth map
        """
        scaled_disp, depth_pred = disp_to_depth(disp, 0.1, 100.0)
        factor = self.get_factor(depth_pred)
        depth_pred *= factor
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        return depth_pred
    
    def get_factor(self, depth: torch.tensor):
        """
        @param disp: depth map, [B, 1, H, W]
        :returns depth factor
        """
        batch_size, _, height, width = depth.shape
        
        # construct intrinsic camera matrix
        intrinsic = self.intrinsic.copy()
        intrinsic[0, :] *= width
        intrinsic[1, :] *= height
        intrinsic = torch.tensor(intrinsic).repeat(batch_size, 1, 1)

        # get camera coordinates
        cam_coords = pixel2cam(depth.squeeze(1), intrinsic.inverse())
        
        # get some samples from the ground, center of the image
        samples = cam_coords[:, 1, height-10:height, width//2 - 50:width//2 + 50]
        samples = samples.reshape(samples.shape[0], -1)
        
        # get the median
        median = samples.median(1)[0]
  
        # get depth factor
        factor = self.CAM_HEIGHT / median
        return factor.reshape(factor.shape, 1, 1, 1)


class Inapint(object):
    def __init__(self, model_name: str, root_dir: str="."):
        self.model_name = model_name
        self.root_dir = root_dir
        self.inpaint = PConvUNet().cuda()
        start_iter = load_ckpt(
            os.path.join(root_dir, "models", model_name, "unet.pth"), 
            [('model', self.inpaint)],
            None
        )
        self.inpaint = self.inpaint.to(device)
        self.inpaint.eval()
        
    def forward(self, img: torch.tensor, mask: torch.tensor):
        """
        @param img: RGB image, [B, 3, H, W]
        @param mask: image mask, [B, 3, H, W]
        :returns torch.tensor [B, 3, H, W]
        """
        if img.max() > 1:
            img /= 255.
        
        img = normalize(img)
        
        # send to gpu
        img = img.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            output, _ = self.inpaint(img, mask)
        
        output = output.cpu()
        output = unnormalize(output)
        
        img = img.cpu()
        img = unnormalize(img)
        mask = mask.cpu()
        
        return output, mask * img + (1 - mask) * output


class Transformation(object):
    def __init__(self):
        self.intrinsic = torch.tensor([
            [0.61, 0, 0.5, 0],   # width
            [0, 1.22, 0.5, 0],   # height
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=torch.float64).unsqueeze(0)
        # self.extrinsic = None 
        self. extrinsic = torch.tensor([
        	[1,  0, 0, 0.00],
        	[0, -1, 0, 1.65],
        	[0,  0, 1, 1.54],
        	[0 , 0, 0, 1.00]], dtype=torch.float64).unsqueeze(0).inverse()
    
    def forward(self, img: torch.tensor, depth: torch.tensor, tx: float = 0.0, ry: float = 0.0):
        """
        @param img: rgb image, [B, 3, H, W]
        @param depth: depth map, [B, 1, H, W]
        @param tx: translation Ox [m]
        @param ry: rotation Oy [rad]
        :returns projected image, mask of valid points
        """
        # casting
        img = img.double()
        depth = depth.double()
        
        batch_size, _, height, width = img.shape
        
        # modify intrinsic
        _, _, height, width = img.shape
        intrinsic = self.intrinsic.clone()
        intrinsic[:, 0] *= width
        intrinsic[:, 1] *= height
        
        # add pose
        pose = torch.zeros((batch_size, 6), dtype=torch.float64)
        pose[:, 0], pose[:, 4] = tx, ry
    
        # down sample
        down = nn.AvgPool2d(2)
        down_img = down(img)
        down_depth= down(depth)

        S = torch.tensor([
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.double)
        intrinsic = torch.matmul(S, intrinsic)
    
        # apply perspective transformation
        projected_img, valid_points = forward_warp(
            img=down_img,
            depth=down_depth.squeeze(1), 
            pose=pose, 
            intrinsics=intrinsic[:, :3, :3].repeat(1, 1, 1),
            extrinsics=self.extrinsic
        )
        
        return down_img, projected_img, valid_points.repeat(1, 3, 1, 1)


if __name__ == "__main__":
    monodepth = Monodepth("monodepth")
    inpaint = Inapint("inpaint")
    transf = Transformation()

    # read and resize image
    img = pil.open("/home/robert/Desktop/Proiect/upb_raw/01fd5e96d7134f50-0/12.png")
    img = img.resize((512, 256))
    img = np.asarray(img)

    # transform image to tensor
    timg = img.transpose(2, 0, 1)
    timg = torch.tensor(timg).unsqueeze(0).float()

    # predict depth
    tdisp = monodepth.forward(timg)
    tdepth = monodepth.get_depth(tdisp)

    # shift image
    timg, tproj_img, tvalid_pts = transf.forward(timg, tdepth, tx=0.0, ry=0.1)

    img = timg.squeeze(0).numpy().transpose(1, 2, 0)
    img = img.astype(np.uint32)
    proj_img = tproj_img.squeeze(0).numpy().transpose(1, 2, 0)
    proj_img = proj_img.astype(np.uint32)

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].imshow(proj_img)
    ax[1].imshow(img)
    plt.show()

    # inpaint
    _, toutput = inpaint.forward(tproj_img.float(), tvalid_pts.float())
    output = toutput.squeeze(0).numpy().transpose(1, 2, 0)
    output = output.astype(np.float)

    # plot final image
    plt.imshow(output)
    plt.show()
