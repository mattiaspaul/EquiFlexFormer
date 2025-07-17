import torch
import os
import time,sys
import argparse
from transformers import ViTModel
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm,trange

import logging
#logging.disable(logging.WARNING)
from torchvision.models import resnet50,resnet34
from torchvision.models._utils import IntermediateLayerGetter

def roto_patch(input):
    input_equi = input.unsqueeze(0).repeat(4,1,1,1,1)
    for i,f in enumerate(((),(-1,),(-2,-1),(-2))):
        input_equi[i] = input.flip(f)
    return input_equi.flatten(0,1)

def equi_flatten(x):
    x = x.unflatten(0,(4,-1))
    x_equi = torch.zeros_like(x)
    for i,f in enumerate(((),(-1,),(-2,-1),(-2))):
        x_equi[i] = x[i].flip(f)
    return x_equi.max(0).values 
    
    
def get_resnet34_stride8_backbone():
    model = resnet34(pretrained=True)

    # Modify layer3 and layer4 to avoid further downsampling
    for n, m in model.layer3.named_modules():
        if 'conv1' in n:
            m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        if 'downsample.0' in n:
            m.stride = (1, 1)
    for n, m in model.layer4.named_modules():
        if 'conv1' in n:
            m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
        if 'downsample.0' in n:
            m.stride = (1, 1)

    # Extract final feature map at 1/8 resolution
    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(model, return_layers=return_layers)
    return backbone
def get_resnet50_stride8_backbone():
    backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
    # Extract feature map from layer4 (1/8 resolution after dilation modifications)
    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return backbone
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
class FlipModel(nn.Module):
    def __init__(self,model1,HW) -> None:
        super().__init__()
        self.model1 = model1
        self.HW = HW
        
    def forward(self, x,interpolate_pos_encoding=True):
        HW = self.HW
        
        embed = self.model1.encoder.layer[0](self.model1.embeddings(x,interpolate_pos_encoding=interpolate_pos_encoding))[0]
        embed = self.model1.encoder.layer[1](embed)[0]
        for ff in zip([[2],[3],[2,3]],[[1],[2],[1,2]]):
            embed2 = self.model1.encoder.layer[0](self.model1.embeddings(x.flip(ff[0]),interpolate_pos_encoding=interpolate_pos_encoding))[0]
            embed2 = self.model1.encoder.layer[1](embed2)[0]
            embed2[:,1:] = embed2[:,1:].unflatten(1,(HW,HW)).flip(ff[1]).flatten(1,2)
            embed = torch.maximum(embed,embed2)
        for i in range(2,len(self.model1.encoder.layer)):
            embed = self.model1.encoder.layer[i](embed)[0]
        y = self.model1.layernorm(embed)[:,1:].permute(0,2,1).unflatten(2,(HW,HW))
        
        return y
def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)
class roto_conv(nn.Module):
    def __init__(self,conv1) -> None:
        super().__init__()
        self.conv1 = conv1

    def forward(self, x):
        return equi_flatten(self.conv1(roto_patch(x)))

import torch
import torch.nn as nn

def low_rank_approximation(linear_layer, rank):
    W = linear_layer.weight.data  # shape: (out_dim, in_dim)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    U_k = U[:, :rank]            # (out_dim, rank)
    S_k = S[:rank]               # (rank,)
    V_k = Vh[:rank, :]           # (rank, in_dim)

    # Construct low-rank approximation: input → rank → output
    down_proj = nn.Linear(V_k.shape[1], rank, bias=False)   # in_dim → rank
    up_proj = nn.Linear(rank, U_k.shape[0], bias=True)      # rank → out_dim

    # Set weights correctly
    down_proj.weight.data = V_k                             # (rank, in_dim)
    
    up_proj.weight.data = (U_k * S_k)#.T                     # (out_dim, rank)

    if linear_layer.bias is not None:
        up_proj.bias.data = linear_layer.bias.data.clone()

    return nn.Sequential(down_proj, up_proj)


def main(args):
    model = args.model
    gpu_num = args.gpu_num
    dataset = args.dataset
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True


    model1 = ViTModel.from_pretrained('facebook/dino-vits8',weights_only=False).cuda()
    
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


    if('dino' in model):
        model1 = ViTModel.from_pretrained('facebook/dino-vits8',weights_only=False).cuda()
        for i in range(0,len(model1.encoder.layer)):
            model1.encoder.layer[i].intermediate.dense = low_rank_approximation(model1.encoder.layer[i].intermediate.dense, 128)
            model1.encoder.layer[i].output.dense = low_rank_approximation(model1.encoder.layer[i].output.dense,128)      
        model2 = Decoder2(num_classes,384,8).cuda()#768).cuda()
        if('reinit' in args.note):
            print('reinitialising global attention')
            for i in range(len(model1.encoder.layer)):
                model1.encoder.layer[i].attention.attention.apply(model1._init_weights)
    elif('resnet' in model):
        model1 = get_resnet34_stride8_backbone().cuda()
        model2 = Decoder2(num_classes,512,8).cuda()
    else:
        print('model not found')
        sys.exit()
        #
    if('flip' in args.model):
        print('using flip model')
        if('resnet' in model):
            if('equi' in model):
                print('using equi flip model')
                for name, module in model1.named_modules():
                    if isinstance(module,nn.Conv2d):
                        before = get_layer(model1, name)
                        if(before.kernel_size[0]>1):
                            set_layer(model1, name, roto_conv(before))
            else:
                print('resnet base flip model online in fwd pass defined ')
        else:
            model1 = FlipModel(model1,HW)

    #model2 = Decoder2(num_classes,768,14.4).cuda()#768).cuda()
    model2 = torch.compile(model2)
    #model2.load_state_dict(state2)

    #model1 = torch.compile(model1)

    #
    #HW = 36#20

    #optimizer = torch.optim.Adam(list(model1.embeddings.patch_embeddings.parameters()),lr=0.0001)

    optimizer = torch.optim.Adam(list(model2.parameters())+list(model1.parameters()),lr=0.0001)
    #print('attention linear probe only')
    num_iterations = 5000#6000#12000
    schedulers = [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0,\
                        total_iters=300),torch.optim.lr_scheduler.StepLR(optimizer,6*2,0.98)]; 
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=[300,])
    t0 = time.time()
    run_loss = torch.zeros(num_iterations,2)

    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):
            optimizer.zero_grad()
            idx = torch.randperm(num_train,device='cuda')[:12].cpu()#1536
            x = imgs[idx].cuda().unsqueeze(1).repeat(1,3,1,1).float()
            labels_gt = labels[idx].long().cuda()

            if('aff' in args.note):
                with torch.no_grad():
                    affine = F.affine_grid(torch.eye(2,3,device='cuda').unsqueeze(0)+torch.randn(12,2,3,device='cuda')*0.07, x.shape, align_corners=False)
                    x = F.grid_sample(x,affine,align_corners=False)
                    labels_gt = F.grid_sample(labels_gt.unsqueeze(1).float(),affine,mode='nearest',align_corners=False).squeeze(1).long()
            if('era' in args.note):
                x = erase_fn(x)
                
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                if('dino' in model):
                    input = model1(x,interpolate_pos_encoding=True)
                    if(hasattr(input, "last_hidden_state")):
                        input = input['last_hidden_state'][:,1:].permute(0,2,1).unflatten(2,(HW,HW))

                    output = model2(input)
                elif('resnet' in model):
                    if(('flip' in model)&(not('equi' in model))):
                        input = equi_flatten(model1(roto_patch(x))['out'])
                    else:
                        input = model1(x)['out']
                    output = model2(input)                
                loss = nn.CrossEntropyLoss()(output,labels_gt.cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
            #run_loss[i,0] = loss.item()
            dice = multiclass_f1_score(output.argmax(1).reshape(-1),labels_gt.reshape(-1).cuda(),num_classes=num_classes,average=None)[1:]
            dice0 = multiclass_f1_score(labels_gt.reshape(-1).cuda(),labels_gt.long().reshape(-1).cuda(),num_classes=num_classes,average=None)[1:]

            run_loss[i,0] = (dice.sum()/dice0.sum()).item()#(output.argmax(1).cpu()==labels[idx]).float().mean().item()

            if(i%10==9):
                with torch.no_grad():
                    idx = torch.randperm(num_val,device='cuda')[:8].cpu()+num_train#633,1536,768m408

                    with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                        #input = imgs[idx].unsqueeze(1).cuda()#feats[idx].cuda()
                        x = imgs[idx].cuda().unsqueeze(1).repeat(1,3,1,1)
                        if('dino' in model):
                            input = model1(x,interpolate_pos_encoding=True)
                            if(hasattr(input, "last_hidden_state")):
                                input = input['last_hidden_state'][:,1:].permute(0,2,1).unflatten(2,(HW,HW))

                            output = model2(input)
                        elif('resnet' in model):
                            if(('flip' in model)&(not('equi' in model))):
                                input = equi_flatten(model1(roto_patch(x))['out'])
                            else:
                                input = model1(x)['out']                            
                        output = model2(input) 

                        dice = multiclass_f1_score( output.argmax(1).reshape(-1),labels[idx].long().reshape(-1).cuda(),num_classes=num_classes,average=None )[1:]
                        dice0 = multiclass_f1_score(labels[idx].long().reshape(-1).cuda(),labels[idx].long().reshape(-1).cuda(),num_classes=num_classes,average=None)[1:]

                        #dice = multiclass_f1_score(output.argmax(1).reshape(-1),labels[idx].long().reshape(-1).cuda(),num_classes=num_classes,average=None)[1:]

                    run_loss[i-9:i+1,1] = (dice.sum()/dice0.sum()).item()


                
            str1 = f"its: {i}, f1-val: {'%0.3f'%(run_loss[i-28:i-1,1].mean())},  f1-train: {'%0.3f'%(run_loss[i-28:i-1,0].mean())}, time: {'%0.3f'%(time.time()-t0)} s, maxVRAM: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GB"
            pbar.set_description(str1)
            pbar.update(1)
            if(i%10==9):
                #torch.save([model.state_dict(),run_loss],f'segresnet_dino_spine.pt')
                torch.save([model1.state_dict(),model2.state_dict(),run_loss],f'sotaLR_'+model+'_dataset_'+dataset+'_'+args.note+'.pt')
            if(i%1000==999):
                torch.save([model1.state_dict(),model2.state_dict(),run_loss],f'sotaLR_'+model+'_dataset_'+dataset+'_'+args.note+'.pt')
                

#label selection chaos     d1 = all_dice1[model+'_'+note][:,torch.tensor([0,1,2,5,6,7,8,9])].view(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'flip DINO on 2D segmentation, args: model, gpu_num, dataset')
    parser.add_argument('model', help='dino, unet, resnet')
    parser.add_argument('gpu_num', help='usually 0-3')
    parser.add_argument('dataset', help='crossmoda, spine, chaos')
    parser.add_argument('note', help='experiment number')
    args = parser.parse_args()

    main(args)