import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from model.image_model import DMCI
from model.video_model import DMC
import torchvision
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import numpy as np


def replicate_pad(x, p):
    height=x.shape[-2]
    width=x.shape[-1]
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    if padding_right == 0 and padding_bottom == 0:
        return x
    return np.pad(x, [(0, 0),(0, 0), (0, padding_bottom), (0, padding_right)], mode='edge')

class Dataset():
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                self.clips.append(full_path)
            
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index):
        clip_path = self.clips[index]
        clip = np.load(clip_path)
        clip=replicate_pad(clip,16)
        clip=clip.astype(np.float32)/255.0
        return clip

q_index_to_lambda = {i: 1 + 24 * (i - 1) for i in range(1, 65)}

def get_distortion(infeat, outfeat, mse):
    distortion = mse(infeat, outfeat)
    return distortion

def get_bpp(x_recon, likelihoods):  # Returns calculated bpp for train and test

    size_est = (-np.log(2) * x_recon.numel())
    bpp = torch.sum(torch.log(likelihoods)) / size_est
    return bpp

'''
def rate_distortion_loss():

    bpp = out_net["bpp_y"] + out_net["bpp_z"]
    out = {"bpp": bpp}
    
    out["mse"] = F.mse_loss(out_net["dpb"]["ref_frame"], target)
    out["psnr"] = 10 * torch.log10(1 * 1 / out["mse"])
    out["loss"] = (self.q_index_to_lambda[q_index] * out["mse"] * self.weights[frame_idx] + out["bpp"])

    return out
'''

def train_one_epoch(i_frame_model, p_frame_model, dataset):


    mse = torch.nn.MSELoss(reduction='mean')
    weights= [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
    for batch in tqdm(dataset):
        batch = batch.to('cuda:0')
        q = random.randint(1, 63) 

        # I frame training
        i_frame_model.train()
        p_frame_model.train()
        recon=None
        feature=None
        loss=0
        optimizer.zero_grad()
        for i in range(0,8):

            if i==0:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = i_frame_model.forward(batch[:, 0], q)
            elif i==1:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, _ = p_frame_model.forward(batch[:, i], recon.to('cuda:0'), None, q)
            else:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, _  = p_frame_model.forward(batch[:, i], None, feature.to('cuda:0'), q)
                
            loss1 = get_bpp(y_hat, y_likelihoods) + get_bpp(z_hat, z_likelihoods) 
            loss2 = get_distortion(batch[:, 0], x_hat, mse)
            loss += q_index_to_lambda[q] * loss2 * weights[i] + loss1
            with torch.no_grad():
                if i==0:
                    i_frame_model.eval()
                    recon, _, _, _, _ = i_frame_model.forward(batch[:, 0], q)
                elif i==1:
                    p_frame_model.eval()
                    _, _, _, _, _, feature = p_frame_model.forward(batch[:, i], recon.to('cuda:0'), None, q)
                else:
                    p_frame_model.eval()
                    _, _, _, _, _, feature = p_frame_model.forward(batch[:, i], None, feature.to('cuda:0'), q)

        loss /= 8;
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_one_epoch(i_frame_model, p_frame_model, dataset, epoch):

    mse = torch.nn.MSELoss(reduction='mean')
    avgmse=0
    avgbpp=0
    i_frame_model.eval()
    p_frame_model.eval()
    recon=None
    feature=None
    count=0
    qpl=[15,31,63]
    with torch.no_grad():
        for batch in tqdm(dataset):
            count+=1
            batch=batch.to('cuda:0')
            q = qpl[count%3]
            for i in range(0,8):
                if i==0:
                    recon, y_hat, y_likelihoods, z_hat, z_likelihoods = i_frame_model.forward(batch[:, 0], q)
                elif i==1:
                    recon, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = p_frame_model.forward(batch[:, i], recon.to('cuda:0'), None, q)
                else:
                    recon, y_hat, y_likelihoods, z_hat, z_likelihoods, feature  = p_frame_model.forward(batch[:, i], None, feature.to('cuda:0'), q)


                avgbpp += get_bpp(y_hat, y_likelihoods) + get_bpp(z_hat, z_likelihoods) 
                avgmse += get_distortion(batch[:, 0], recon, mse)
    avgbpp/=len(dataset)
    avgmse/=len(dataset)
    torch.save(i_frame_model.state_dict(), 'logs/i-Ep{}-dev{:.3f}-{:.3f}.pth'.format(epoch + 1, avgbpp, avgmse))
    torch.save(p_frame_model.state_dict(), 'logs/p-Ep{}-dev{:.3f}-{:.3f}.pth'.format(epoch + 1, avgbpp, avgmse))


if __name__ == '__main__':

    torch.manual_seed(42)
    random.seed(42)
    lr = 1e-4
    Batch_size = 2
    Init_Epoch = 0
    Fin_Epoch = 200
    device = "cuda"
    train_dataset = Dataset('videos/train_clip')
    test_dataset = Dataset('videos/test_clip')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=Batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    

    inet= torch.nn.DataParallel(DMCI(), device_ids=[0, 1])
    pnet= torch.nn.DataParallel(DMC(), device_ids=[0, 1])
    inet = inet.to("cuda:0")
    pnet = pnet.to("cuda:0")

    optimizer = torch.optim.Adam(list(inet.parameters()) + list(pnet.parameters()), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    for epoch in range(Init_Epoch, Fin_Epoch):
        train_one_epoch(inet, pnet, train_dataloader)
        test_one_epoch(inet, pnet, test_dataloader, epoch)
        lr_scheduler.step()


