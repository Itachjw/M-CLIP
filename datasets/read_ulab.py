from utils.iotools import read_image
import torchvision.transforms as T
import torch
import pdb

def read_ulab_image(args, images1, images2, images3, images4, images5, p_uids, ulab_id_path):
    height, width = (384, 128)

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomErasing(scale=(0.02, 0.4), value=mean),
    ])
        
    batch_size = images1.shape[0]
    for i in range(batch_size):
        num = torch.clamp(p_uids[i][-1], 0, 10)
        if num==1:
            img_path1 = ulab_id_path[p_uids[i][0]]
            img1 = transform(read_image(img_path1)).cuda()
            images1[i] = img1
            
        elif num==2:
            img_path1 = ulab_id_path[p_uids[i][1]]
            img1 = transform(read_image(img_path1)).cuda()
            images1[i] = img1
            
            img_path2 = ulab_id_path[p_uids[i][0]]
            img2 = transform(read_image(img_path2)).cuda()
            images2[i] = img2
            
            
        elif num==3:
            img_path1 = ulab_id_path[p_uids[i][2]]
            img1 = transform(read_image(img_path1)).cuda()
            images1[i] = img1
            
            img_path2 = ulab_id_path[p_uids[i][1]]
            img2 = transform(read_image(img_path2)).cuda()
            images2[i] = img2

            img_path3 = ulab_id_path[p_uids[i][0]]
            img3 = transform(read_image(img_path3)).cuda()
            images3[i] = img3
            
        elif num==4:
            img_path1 = ulab_id_path[p_uids[i][3]]
            img1 = transform(read_image(img_path1)).cuda()
            images1[i] = img1
            
            img_path2 = ulab_id_path[p_uids[i][2]]
            img2 = transform(read_image(img_path2)).cuda()
            images2[i] = img2
            
            img_path3 = ulab_id_path[p_uids[i][1]]
            img3 = transform(read_image(img_path3)).cuda()
            images3[i] = img3
            
            img_path4 = ulab_id_path[p_uids[i][0]]
            img4 = transform(read_image(img_path4)).cuda()
            images4[i] = img4
            
        elif num>4:
            img_path1 = ulab_id_path[p_uids[i][num-1]]
            img1 = transform(read_image(img_path1)).cuda()
            images1[i] = img1
            
            img_path2 = ulab_id_path[p_uids[i][num-2]]
            img2 = transform(read_image(img_path2)).cuda()
            images2[i] = img2
            
            img_path3 = ulab_id_path[p_uids[i][num-3]]
            img3 = transform(read_image(img_path3)).cuda()
            images3[i] = img3
            
            img_path4 = ulab_id_path[p_uids[i][num-4]]
            img4 = transform(read_image(img_path4)).cuda()
            images4[i] = img4
            
            img_path5 = ulab_id_path[p_uids[i][num-5]]
            img5 = transform(read_image(img_path5)).cuda()
            images5[i] = img5

    return images1, images2, images3, images4, images5