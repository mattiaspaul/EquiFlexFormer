import torch
import os
import time,sys
import argparse
from transformers import ViTModel
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm,trange
from monai.networks.nets import DynUNet

import logging
#logging.disable(logging.WARNING)

def roto_patch(input):
    input_equi = input.unsqueeze(0).repeat(4,1,1,1,1)
    for i,f in enumerate(((),(-1,),(-2,-1),(-2))):
        input_equi[i] = input.flip(f)
    return input_equi.flatten(0,1)

def equi_flatten_mean(x):
    x = x.unflatten(0,(4,-1))
    x_equi = torch.zeros_like(x)
    for i,f in enumerate(((),(-1,),(-2,-1),(-2))):
        x_equi[i] = x[i].flip(f)
    return x_equi.mean(0)


class roto_conv_mean(nn.Module):
    def __init__(self,conv1) -> None:
        super().__init__()
        self.conv1 = conv1

    def forward(self, x):
        return equi_flatten_mean(self.conv1(roto_patch(x)))


class SuppressMissingClassWarning(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # drop only torcheval warnings about “class … does not exist”
        if "Some classes do not exist in the target" in msg:
            return False
        return True

# install on the root logger
logging.getLogger().addFilter(SuppressMissingClassWarning())

from torcheval.metrics.functional import multiclass_f1_score

from torchvision.transforms import RandomErasing
erase = RandomErasing(p=1.0,inplace=True)
def erase_fn(x):
    x = x.clone()
    for i in range(x.size(0)):
        erase(x[i])
    return x

class Decoder2(nn.Module):
    def __init__(self,max_label,embd_dim=384,scale_factor=8) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(embd_dim,max_label,1)
    def forward(self,x):
        return F.interpolate(self.conv(x),scale_factor=self.scale_factor,mode='bilinear')
    


def main(args):
    model = args.model
    gpu_num = args.gpu_num
    dataset = args.dataset
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    print(model,gpu_num,dataset)

    if(dataset == 'crossmoda'):
        imgs,labels,shapes = torch.load('crossmoda0.pth'); num_classes = 2; num_train = 768; num_val = 408; HW = 28
    elif(dataset=='spine'):
        imgs,labels= torch.load('spine_mr.pth'); num_classes = 20; num_train = 1536; num_val = 633; HW = 36
    elif(dataset=='amos'):
        imgs,labels,shapes = torch.load('amosmr.pth'); num_classes = 14; num_train = 1755; num_val = 720; HW = 40
    else:
        print('dataset not found')
        sys.exit()


    number_of_downsampling_layers = 5
    strides = [2,1] * (number_of_downsampling_layers)
    filters = [64, 64, 96, 96, 128, 128, 192, 192, 256, 256, 384, 384, 512, 512][: len(strides)]
    filters = [filt // 2 for filt in filters]  # half the filters for each layer
    kernel_size = [3,3] * (number_of_downsampling_layers)
    upsample_kernel_size = [2,1] * (number_of_downsampling_layers)
    unet = DynUNet(spatial_dims=2, in_channels=1, out_channels=num_classes, filters=filters, kernel_size=kernel_size, strides=strides, upsample_kernel_size=upsample_kernel_size[1:],deep_supervision=True,deep_supr_num=number_of_downsampling_layers-2).cuda()

    if('flip' in args.model):
        print('unet flip model online in fwd pass defined ')


    optimizer = torch.optim.Adam(list(unet.parameters()),lr=0.001)

    num_iterations = 5000

    t0 = time.time()
    run_loss = torch.zeros(num_iterations,2)

    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):
            optimizer.zero_grad()
            idx = torch.randperm(num_train,device='cuda')[:12].cpu()#1536
            x = imgs[idx].cuda().unsqueeze(1).float()
            labels_gt = labels[idx].long().cuda()

            if('aff' in args.note):
                with torch.no_grad():
                    affine = F.affine_grid(torch.eye(2,3,device='cuda').unsqueeze(0)+torch.randn(12,2,3,device='cuda')*0.07, x.shape, align_corners=False)
                    x = F.grid_sample(x,affine,align_corners=False)
                    labels_gt = F.grid_sample(labels_gt.unsqueeze(1).float(),affine,mode='nearest',align_corners=False).squeeze(1).long()
            if('era' in args.note):
                x = erase_fn(x)
                
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                if(('flip' in model)):
                    input = equi_flatten_mean(unet(roto_patch(x))[:,-1])
                else:
                    input = unet(x)[:,-1]
                output = F.interpolate(input,scale_factor=2,mode='bilinear',align_corners=False)
                loss = nn.CrossEntropyLoss()(output,labels_gt.cuda())
            loss.backward()
            optimizer.step()

            dice = multiclass_f1_score(output.argmax(1).reshape(-1),labels_gt.reshape(-1).cuda(),num_classes=num_classes,average=None)[1:]
            dice0 = multiclass_f1_score(labels_gt.reshape(-1).cuda(),labels_gt.long().reshape(-1).cuda(),num_classes=num_classes,average=None)[1:]

            run_loss[i,0] = (dice.sum()/dice0.sum()).item()#(output.argmax(1).cpu()==labels[idx]).float().mean().item()

            if(i%10==9):
                with torch.no_grad():
                    idx = torch.randperm(num_val,device='cuda')[:8].cpu()+num_train#633,1536,768m408

                    with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                        x = imgs[idx].cuda().unsqueeze(1)
                        if(('flip' in model)):
                            input = equi_flatten_mean(unet(roto_patch(x))[:,-1])
                        else:
                            input = unet(x)[:,-1]
                        output = F.interpolate(input,scale_factor=2,mode='bilinear',align_corners=False)


                        dice = multiclass_f1_score( output.argmax(1).reshape(-1),labels[idx].long().reshape(-1).cuda(),num_classes=num_classes,average=None )[1:]
                        dice0 = multiclass_f1_score(labels[idx].long().reshape(-1).cuda(),labels[idx].long().reshape(-1).cuda(),num_classes=num_classes,average=None)[1:]

                        #dice = multiclass_f1_score(output.argmax(1).reshape(-1),labels[idx].long().reshape(-1).cuda(),num_classes=num_classes,average=None)[1:]

                    run_loss[i-9:i+1,1] = (dice.sum()/dice0.sum()).item()


                
            str1 = f"its: {i}, f1-val: {'%0.3f'%(run_loss[i-28:i-1,1].mean())},  f1-train: {'%0.3f'%(run_loss[i-28:i-1,0].mean())}, time: {'%0.3f'%(time.time()-t0)} s, maxVRAM: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GB"
            pbar.set_description(str1)
            pbar.update(1)
            if(i%10==9):
                #torch.save([model.state_dict(),run_loss],f'segresnet_dino_spine.pt')
                torch.save([unet.state_dict(),run_loss],f'sotaH_'+model+'_dataset_'+dataset+'_'+args.note+'.pt')
            if(i%1000==999):
                torch.save([unet.state_dict(),run_loss],f'sotaH_'+model+'_dataset_'+dataset+'_'+args.note+'.pt')
                

#label selection chaos     d1 = all_dice1[model+'_'+note][:,torch.tensor([0,1,2,5,6,7,8,9])].view(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'dynunet on 2D segmentation, args: model, gpu_num, dataset')
    parser.add_argument('model', help='unet')
    parser.add_argument('gpu_num', help='usually 0-3')
    parser.add_argument('dataset', help='crossmoda, spine, chaos')
    parser.add_argument('note', help='experiment number')
    args = parser.parse_args()

    main(args)