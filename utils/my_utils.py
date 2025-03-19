import torch
import random

def getmask(batch_size ,num_channels ,height ,width ,fraction_to_drop = None ):
    fraction_to_drop = random.uniform(0.0, 0.8) 
    num_pixels = height * width
    num_to_drop = int(num_pixels * fraction_to_drop)  
    mask = torch.ones((batch_size, num_channels, height, width))

    print("LOG:" , fraction_to_drop , num_to_drop)
    for i in range(batch_size):
        idx = torch.randperm(num_pixels)[:num_to_drop] 
        mask[i].view(-1)[idx] = 0 
    return mask

def pruned_gaussians(gaussian_splats ,mask ,batch_size):

    
    gaussian_splats['xyz']= gaussian_splats['xyz'][mask.bool().reshape(batch_size, -1), :].reshape(batch_size, -1, 3)

    gaussian_splats['rotation']= gaussian_splats['rotation'][mask.bool().reshape(batch_size, -1), :].reshape(batch_size, -1, 4)

    gaussian_splats['features_dc']= gaussian_splats['features_dc'][mask.bool().reshape(batch_size, -1), :].reshape(batch_size, -1, 1,3)

    gaussian_splats['opacity']= gaussian_splats['opacity'][mask.bool().reshape(batch_size, -1), :].reshape(batch_size, -1, 1)

    gaussian_splats['scaling']= gaussian_splats['scaling'][mask.bool().reshape(batch_size, -1), :].reshape(batch_size, -1, 3)

    gaussian_splats['features_rest']= gaussian_splats['features_rest'][mask.bool().reshape(batch_size, -1), :].reshape(batch_size, -1, 3 ,3)

    return gaussian_splats
