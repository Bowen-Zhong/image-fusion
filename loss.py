








import torch.nn.functional as F
import torch
from math import exp
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def compute_loss(fusion, img_cat, img_1, img_2):
#
#     weight = [1, 1, 1, 2.5]
#     loss_ssim_ir = weight[0] * ssim_ir(fusion, img_1)
#     loss_ssim_vi = weight[1] * ssim_vi(fusion, img_2)
#     loss_RMI_ir = weight[2] * RMI_ir(fusion, img_1)
#     loss_RMI_vi = weight[3] * RMI_vi(fusion, img_2)

        # return loss_ssim_ir + loss_ssim_vi + loss_RMI_ir + loss_RMI_vi
    # return loss_ssim_ir + loss_ssim_vi
def RMI_vi (input_vi,fused_result ):
    RMI_vi=RMI(input_vi,fused_result)

    return RMI_vi
def RMI_ir (input_ir,fused_result ):
    RMI_ir=RMI(input_ir,fused_result)

    return RMI_ir

def ssim_vi (fused_result,input_vi ):
    ssim_vi=ssim(fused_result,input_vi)

    return ssim_vi

def ssim_ir (fused_result,input_ir ):
    ssim_ir=ssim(fused_result,input_ir)

    return ssim_ir


def ssim_loss (fused_result,input_ir,input_vi ):
    ssim_loss=ssim(fused_result,torch.maximum(input_ir,input_vi))

    return ssim_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)


    ret = ssim_map.mean()

    return 1-ret

EPSILON = 0.0005

def RMI(x,y):
    RMI =RMILoss().to(device)
    RMI = RMI(x,y)

    return RMI

class RMILoss(nn.Module):

    def __init__(self,
                 with_logits=True,
                 radius=3,
                 bce_weight=0.5,
                 downsampling_method='max',
                 stride=3,
                 use_log_trace=True,
                 use_double_precision=True,
                 epsilon=EPSILON):

        super().__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, input, target):

        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(input, target=target)
            else:
                bce = F.binary_cross_entropy(input, target=target)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        if self.with_logits:
            input = torch.sigmoid(input)

        rmi = self.rmi_loss(input=input, target=target)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

    def rmi_loss(self, input, target):


        assert input.shape == target.shape
        vector_size = self.radius * self.radius

        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input)

        if self.use_double_precision:
            y = y.double()
            p = p.double()

        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        y_cov = y @ transpose(y)
        p_cov = p @ transpose(p)
        y_p_cov = y @ transpose(p)

        m = y_cov - y_p_cov @ transpose(inverse(p_cov + eps)) @ transpose(y_p_cov)

        if self.use_log_trace:
            rmi = 0.5 * log_trace(m + eps)
        else:
            rmi = 0.5 * log_det(m + eps)

        rmi = rmi / float(vector_size)

        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):


        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == 'region-extraction' else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius ** 2, -1))
        return x_regions

    def downsample(self, x):

        if self.stride == 1:
            return x

        if self.downsampling_method == 'region-extraction':
            return x

        padding = self.stride // 2
        if self.downsampling_method == 'max':
            return F.max_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        if self.downsampling_method == 'avg':
            return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        raise ValueError(self.downsampling_method)


def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x):
    x = torch.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def log_det(x):
    return torch.logdet(x)





import torch

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_loss(fusion, img_cat, img_1, img_2):
    #weight = [0.03, 1000, 10, 100]
    weight = [10, 1000, 10, 100]
    Loss_IR = weight[0] * CharbonnierLoss_IR(fusion, img_1)
    Loss_VI = weight[1] * CharbonnierLoss_VI(fusion, img_2)
    loss_tv_ir = weight[2] * tv_ir(fusion, img_1)
    loss_tv_vi = weight[3] * tv_vi(fusion, img_2)
    loss = Loss_IR + Loss_VI + loss_tv_ir + loss_tv_vi

    return loss

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def tv_loss(x):
    tv_loss =TVLoss().to(device)
    tv_loss = tv_loss(x)

    return tv_loss

def tv_vi (fused_result,input_vi ):
    tv_vi=torch.norm((tv_loss(fused_result)-tv_loss(input_vi)),1)

    return tv_vi


def tv_ir (fused_result,input_ir):
    tv_r=torch.norm((tv_loss(fused_result)-tv_loss(input_ir)),1)

    return tv_r

def CharbonnierLoss_IR(f,ir):
    eps = 1e-3
    loss=torch.mean(torch.sqrt((f-ir)**2+eps**2))
    return loss

def CharbonnierLoss_VI(f,vi):
    eps = 1e-3
    loss=torch.mean(torch.sqrt((f-vi)**2+eps**2))
    return loss

