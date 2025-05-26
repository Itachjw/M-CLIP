import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import pdb

def patch_mix(images, patch_size=16):
    h_n = int(images.shape[2]/patch_size)
    w_n = int(images.shape[3]/patch_size)
    num = int(h_n*w_n)
    mask = torch.cat((torch.ones(int(num/2)), torch.zeros(int(num/2))), 0)
    index = torch.randperm(mask.shape[0])
    mask = mask[index]
    
    img = torch.ones(num, patch_size, patch_size)
    img = img*(mask.unsqueeze(-1).unsqueeze(-1))
    img = rearrange(img, '(h w) s1 s2 -> h w s1 s2', h=h_n)
    img = rearrange(img, 'h w s1 s2 -> h s1 w s2')
    img = rearrange(img, 'h s1 w s2 -> (h s1) (w s2)')
    img = img.unsqueeze(0).unsqueeze(0).cuda()
    
    #pdb.set_trace()
    #b = int(images.shape[0]/2)
    images_flip = torch.flip(images, [0])
    mix_img = images*img + images_flip*(1-img)
    
    return mix_img

def p_match_loss(image_fetures, text_fetures, text_fetures1, text_fetures2, text_fetures3, pid, logit_scale, epsilon=1e-8):
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    label = (pid_dist == 0).float()

    text_fetures = torch.cat((text_fetures, text_fetures1, text_fetures2, text_fetures3), dim=0)   # batch*4, dim
    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = logit_scale*text_norm @ image_norm.t()  ## batch*4   batch  
    #sim_matrix = F.softmax(logit_scale*t2i_cosine_theta, dim=1)*F.softmax(logit_scale*t2i_cosine_theta, dim=0)

    labels = torch.cat((0.4*label, 0.2*label, 0.2*label, 0.2*label), dim=0)   # batch*4, batch
    #pdb.set_trace()
    pred = F.softmax(t2i_cosine_theta, dim=0)
    loss = pred * (F.log_softmax(t2i_cosine_theta, dim=0) - torch.log(labels + epsilon))
    loss= torch.mean(torch.sum(loss, dim=0))
    
    return loss

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_sdm_mix(image_fetures, text_fetures, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    labels1 = torch.eye(batch_size)
    labels2 = torch.flip(labels1, [0])
    labels = (labels1 + labels2).to(image_fetures.device)
    
    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    #pdb.set_trace()
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)

def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

def dual_softmax_loss(image_fetures, text_fetures, pid, logit_scale, epsilon=1e-8):
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    sim_matrix = F.softmax(logit_scale*t2i_cosine_theta, dim=1)*F.softmax(logit_scale*t2i_cosine_theta, dim=0)

    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_t =F.cross_entropy(sim_matrix.t(), labels)
    loss = (loss_i +  loss_t)/2

    return loss, sim_matrix, t2i_cosine_theta

def hard_loss(sim1, sim2, sim3, pid, logit_scale):
    
    batch_size = sim1.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    pos = torch.min(torch.min(sim1, sim2), sim3)
    neg = torch.max(torch.max(sim1, sim2), sim3)

    #pdb.set_trace()
   
    t2i_cosine_theta = labels*pos + (1-labels)*neg
    #sim_matrix = F.softmax(logit_scale*t2i_cosine_theta, dim=1)*F.softmax(logit_scale*t2i_cosine_theta, dim=0)

    loss_i = F.cross_entropy(t2i_cosine_theta, labels)
    loss_t =F.cross_entropy(t2i_cosine_theta.t(), labels)
    loss = (loss_i +  loss_t)/2 
   
    return loss

def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss


def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss

def compute_TRL_per(scores, pid, margin = 0.2, tau=0.02):       
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2

 
def compute_InfoNCE_per(scores, logit_scale):
    
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log())/2    
    return loss

def cross_cent_loss(xx, yy, pid, logit_scale):
    batch_size = xx.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    num = xx.shape[1]
    labels = repeat(labels.unsqueeze(0), 'n h w -> (n rep) h w', rep=num)
    x_cent = xx.mean(dim=1)
    y_cent = yy.mean(dim=1)
    
    x_cent = x_cent / x_cent.norm(dim=-1, keepdim=True)
    y_cent = y_cent / y_cent.norm(dim=-1, keepdim=True)
    xx_norm = xx / xx.norm(dim=-1, keepdim=True)
    yy_norm = yy / yy.norm(dim=-1, keepdim=True)
    
    for i in range(num):
        xc_sim = x_cent @ yy_norm[:,i,:].t()
        yc_sim = y_cent @ xx_norm[:,i,:].t()
        if i==0:
            x2y_sim = xc_sim.unsqueeze(0)
            y2x_sim = yc_sim.unsqueeze(0)
        else:
            x2y_sim = torch.cat((x2y_sim, xc_sim.unsqueeze(0)), 0)
            y2x_sim = torch.cat((y2x_sim, yc_sim.unsqueeze(0)), 0)
            
    x2y_sim = F.softmax(logit_scale*x2y_sim, dim=1)*F.softmax(logit_scale*x2y_sim, dim=0)
    y2x_sim = F.softmax(logit_scale*y2x_sim, dim=1)*F.softmax(logit_scale*y2x_sim, dim=0)
    
    #pdb.set_trace()
    
    loss_i = F.cross_entropy(x2y_sim, labels)
    loss_t = F.cross_entropy(y2x_sim, labels)
    loss = (loss_i +  loss_t)/2
    
    return loss.mean()

def compute_TAL_per1(i_feats, t_feats, rep, batch_size, tau=0.02, margin=0.5):

    pid = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    pid = pid.to(i_feats.device)
    pid = pid.repeat(rep)
    #pdb.set_trace()
    pid = pid.reshape((batch_size*rep, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    i_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
    t_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)
    scores = t_norm @ i_norm.t()

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss.mean() 

def compute_TAL_per(i_feats, t_feats, t_feats1, t_feats2, t_feats3, tau=0.02, margin=0.5):
    #tau = 0.02
    #margin = 0.5
    batch_size = i_feats.shape[0]
    labs = torch.eye(batch_size)
    labs = labs.to(i_feats.device)
    labels = torch.cat((labs, labs, labs, labs), 0)
    mask = 1 - labels
    
    logit_scale = 1 / 0.02
    
    t_emb = torch.cat((t_feats, t_feats1, t_feats2, t_feats3), 0)
    i_emb = i_feats
    
    i_norm = i_emb / i_emb.norm(dim=-1, keepdim=True)
    t_norm = t_emb / t_emb.norm(dim=-1, keepdim=True)
    scores = t_norm @ i_norm.t()
    
    #pdb.set_trace()
    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels.t() / ((scores.t()/tau).exp()* labels.t()).sum(dim=1, keepdim=True)).detach()
    
    loss1 = ( - (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    loss2 = (-  (alpha_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask.t()).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss1.mean() + loss2.mean()

def compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, pid, label_hat=None, tau=0.02, margin=0.1, loss_type='TAL', logit_scale=50):

    loss_bgm, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale)
    loss_tse, _ = compute_per_loss(i_tse_f, t_tse_f, pid, tau, margin, loss_type, logit_scale)

    loss_bgm = (label_hat*loss_bgm).sum()
    loss_tse = (label_hat*loss_tse).sum()
    
    if loss_type in ['TAL','TRL']:
        return loss_bgm, loss_tse
    else:
        return loss_bgm/label_hat.sum(), loss_tse/label_hat.sum() # mean

def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    else:
        exit()

    return per_loss, scores.diag()
    
def local_global_alignment(x, x_l, logit_scale, logit_weight):
    ### x    batch   dim
    ### x_l  batch k dim
    
    batch_size = x.shape[0]
    #x = x/x.norm(dim=-1, keepdim=True)
    #x_l = x_l/x_l.norm(dim=-1, keepdim=True)
    
    xx = repeat(x, 'h w -> (h repeat) w', repeat=batch_size)
    xx = rearrange(xx, '(a b) w -> a b w', a=batch_size)
    
    logits = logit_scale * torch.sum(torch.matmul(xx, x_l.permute(0,2,1)) * torch.matmul(torch.softmax(torch.matmul(xx, x_l.permute(0,2,1)) / 0.01, dim=-1), logit_weight), dim = -1).t()

    pdb.set_trace()
    return logits.squeeze(0)


def triplet_hard_loss(anc, pos, neg, margin):
    
    anc_norm = anc / anc.norm(dim=-1, keepdim=True)
    pos_norm = pos / pos.norm(dim=-1, keepdim=True)
    neg_norm = neg / neg.norm(dim=-1, keepdim=True)
    
    s_pos = torch.sum(anc_norm * pos_norm, dim=-1)
    s_neg = torch.sum(pos_norm * neg_norm, dim=-1)
    
    loss = F.relu(s_neg - s_pos + margin)
    
    return loss.mean()