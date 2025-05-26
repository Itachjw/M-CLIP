from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .text_feature_extract import TextExtract
from utils.simple_tokenizer import SimpleTokenizer
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from datasets.read_ulab import read_ulab_image
from torchvision.models.feature_extraction import create_feature_extractor
import pdb

class CrossAtt(nn.Module):
    def __init__(self,
                 embed_dim=512):
        super(CrossAtt, self).__init__()
        self.embed_dim = embed_dim
        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
    def forward(self, x, y):
        xx = self.cross_attn(
                self.ln_pre_t(x),
                self.ln_pre_i(y),
                self.ln_pre_i(y),
                need_weights=False)[0]
        return xx

class DiscrepLearning(nn.Module):
    def __init__(self, args, embed_dim):
        super(DiscrepLearning, self).__init__()
        
    def forward(self, x, y):
    
        x_norm = x / x.norm(dim=1, keepdim=True)
        y_norm = y / y.norm(dim=1, keepdim=True)
        
        y2x_sim = y_norm @ x_norm.transpose(2,1)
        r_sim = 1. - F.softmax(y2x_sim, dim=-1)
        feats = r_sim @ x #+ x
        
        return feats

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        #self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        self.CAtt = CrossAtt(self.embed_dim)
        self.DPL = DiscrepLearning(self.args, self.embed_dim)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, epoch, k_uids, ulab_id_path):
        ret = dict()

        images = batch['images']
        images1 = batch['images1']
        images2 = batch['images2']
        images3 = batch['images3']
        images4 = batch['images4']
        images5 = batch['images5']
        caption_ids1 = batch['mlm_ids1']
        caption_ids2 = batch['mlm_ids2']
        caption_ids3 = batch['mlm_ids3']
        caption_ids4 = batch['mlm_ids4']
        caption_ids5 = batch['mlm_ids5']
        cap_ids = batch['caption_ids']
        
        pids = batch['pids']
        p_uids = k_uids[pids]
        images1, images2, images3, images4, images5 = read_ulab_image(self.args, images1, images2, images3, images4, images5, p_uids, ulab_id_path)
        #pdb.set_trace()
        
        index = torch.minimum(torch.ones_like(cap_ids.argmax(dim=-1))*76, cap_ids.argmax(dim=-1)+1)
        image_feats, image_m, text_feats, text_m = self.base_model(images, cap_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), index].float()

        image_feats1, _ = self.base_model.encode_image(images1)
        i_feats1 = image_feats1[:, 0, :].float()
        text_feats1, _ = self.base_model.encode_text(cap_ids)#caption_ids1)
        t_feats1 = text_feats1[torch.arange(text_feats1.shape[0]), index].float()
        
        image_feats2, _ = self.base_model.encode_image(images2)
        i_feats2 = image_feats2[:, 0, :].float()
        text_feats2, _ = self.base_model.encode_text(cap_ids)#caption_ids2)
        t_feats2 = text_feats2[torch.arange(text_feats2.shape[0]), index].float()

        #'''  
        image_feats3, _ = self.base_model.encode_image(images3)
        i_feats3 = image_feats3[:, 0, :].float()
        text_feats3, _ = self.base_model.encode_text(cap_ids)#caption_ids3)
        t_feats3 = text_feats3[torch.arange(text_feats3.shape[0]), index].float()
        
        '''        
        image_feats4, _ = self.base_model.encode_image(images4)
        i_feats4 = image_feats4[:, 0, :].float()
        text_feats4, _ = self.base_model.encode_text(caption_ids4)
        t_feats4 = text_feats4[torch.arange(text_feats4.shape[0]), index].float()

        image_feats5, _ = self.base_model.encode_image(images5)
        i_feats5 = image_feats5[:, 0, :].float()
        text_feats5, _ = self.base_model.encode_text(caption_ids5)
        t_feats5 = text_feats5[torch.arange(text_feats5.shape[0]), index].float()
        '''
        
        #pdb.set_trace()

        logit_scale = 1 / 0.02 #self.logit_scale
            
        ret.update({'temperature': 1 / logit_scale})

        #pdb.set_trace()
        if 'itc' in self.current_task:

            itc_loss = objectives.compute_itc(i_feats, t_feats, logit_scale)
            #itc_loss = objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)
            loss = itc_loss
            ret.update({'g_sdm_loss':loss})
            
        if 'sdm' in self.current_task:

            #sdm_loss = objectives.compute_sdm(i_feats, t_feats1, batch['pids'], logit_scale)
            #f_loss00 = objectives.compute_itc(i_feats, t_feats, logit_scale)           
            #f_loss10 = objectives.compute_itc(i_feats1, t_feats1, logit_scale)
            #f_loss20 = objectives.compute_itc(i_feats2, t_feats2, logit_scale)
            #f_loss30 = objectives.compute_itc(i_feats3, t_feats3, logit_scale)
            
            f_loss00, matrix00, cosine00 = objectives.dual_softmax_loss(i_feats, t_feats, batch['pids'], logit_scale)
            f_loss10, matrix10, cosine10 = objectives.dual_softmax_loss(i_feats1, t_feats1, batch['pids'], logit_scale)
            f_loss20, matrix20, cosine20 = objectives.dual_softmax_loss(i_feats2, t_feats2, batch['pids'], logit_scale)
            f_loss30, matrix30, cosine30 = objectives.dual_softmax_loss(i_feats3, t_feats3, batch['pids'], logit_scale)
            #f_loss40, matrix40, cosine40 = objectives.dual_softmax_loss(i_feats4, t_feats4, batch['pids'], logit_scale)
            #f_loss50, matrix50, cosine50 = objectives.dual_softmax_loss(i_feats5, t_feats5, batch['pids'], logit_scale)
            
            #i_f = torch.cat((i_feats, i_feats1, i_feats2, i_feats3, i_feats4, i_feats5), 0)
            #t_f = torch.cat((t_feats, t_feats1, t_feats2, t_feats3, t_feats4, t_feats5), 0)
            #tal_loss = objectives.compute_TAL_per1(i_f, t_f, rep=6, batch_size=i_feats1.shape[0], tau=0.02, margin=0.5)

            i_f = torch.cat((i_feats, i_feats1, i_feats2, i_feats3), 0)
            t_f = torch.cat((t_feats, t_feats1, t_feats2, t_feats3), 0)
            tal_loss = objectives.compute_TAL_per1(i_f, t_f, rep=4, batch_size=i_feats1.shape[0], tau=0.02, margin=0.5)
            
            #i_f = torch.cat((i_feats, i_feats1, i_feats2), 0)
            #t_f = torch.cat((t_feats, t_feats1, t_feats2), 0)
            #tal_loss = objectives.compute_TAL_per1(i_f, t_f, rep=3, batch_size=i_feats1.shape[0], tau=0.02, margin=0.5)
            
            #i_f = torch.cat((i_feats, i_feats1), 0)
            #t_f = torch.cat((t_feats, t_feats1), 0)
            #tal_loss = objectives.compute_TAL_per1(i_f, t_f, rep=2, batch_size=i_feats1.shape[0], tau=0.02, margin=0.5)
            
            loss = f_loss00 + f_loss10 + f_loss20 + tal_loss #+ f_loss30 + f_loss40 + f_loss50
            
            ret.update({'f_sdm_loss':loss})
            
        if 'cmpm' in self.current_task:

            cross_feats = self.CAtt(text_feats, image_feats)
            cross_feats1 = self.CAtt(text_feats1, image_feats1)
            cross_feats2 = self.CAtt(text_feats2, image_feats2)
            cross_feats3 = self.CAtt(text_feats3, image_feats3)
            #cross_feats4 = self.CAtt(text_feats4, image_feats4)
            #cross_feats5 = self.CAtt(text_feats5, image_feats5)
            
            margin = 0.2
            att_words1 = self.DPL(cross_feats, cross_feats1)
            att_words2 = self.DPL(cross_feats, cross_feats2) 
            att_words3 = self.DPL(cross_feats, cross_feats3)   
            #att_words4 = self.DPL(cross_feats, cross_feats4)   
            #att_words5 = self.DPL(cross_feats, cross_feats5)               
            
            text_fs1, _ = self.base_model.decode_text(batch['mlm_ids1'], att_words1, batch['mlm_labels1'])
            #text_fs1, _ = self.base_model.decode_text1(batch['mlm_ids1'], att_words1)
            t_fs1 = text_fs1[torch.arange(text_feats.shape[0]), index].float()
            loss1 = objectives.triplet_hard_loss(t_fs1, i_feats.detach(), t_feats1.detach(), margin)
            
           
            text_fs2, _ = self.base_model.decode_text(batch['mlm_ids2'], att_words2, batch['mlm_labels2'])
            #text_fs2, _ = self.base_model.decode_text1(batch['mlm_ids2'], att_words2)
            t_fs2 = text_fs2[torch.arange(text_feats.shape[0]), index].float()
            loss2 = objectives.triplet_hard_loss(t_fs2, i_feats.detach(), t_feats2.detach(), margin)
            
            #'''
            text_fs3, _ = self.base_model.decode_text(batch['mlm_ids3'], att_words3, batch['mlm_labels3'])
            #text_fs3, _ = self.base_model.decode_text1(batch['mlm_ids3'], att_words3)
            t_fs3 = text_fs3[torch.arange(text_feats.shape[0]), index].float()
            loss3 = objectives.triplet_hard_loss(t_fs3, i_feats.detach(), t_feats3.detach(), margin)
            
            '''            
            text_fs4, _ = self.base_model.decode_text(batch['mlm_ids4'], att_words4, batch['mlm_labels4'])
            #text_fs4, _ = self.base_model.decode_text1(batch['mlm_ids4'], att_words4)
            t_fs4 = text_fs4[torch.arange(text_feats.shape[0]), index].float()
            loss4 = objectives.triplet_hard_loss(t_fs4, i_feats.detach(), t_feats4.detach(), margin)
            
            text_fs5, _ = self.base_model.decode_text(batch['mlm_ids5'], att_words5, batch['mlm_labels5'])
            #text_fs5, _ = self.base_model.decode_text1(batch['mlm_ids5'], att_words5)
            t_fs5 = text_fs5[torch.arange(text_feats.shape[0]), index].float()
            loss5 = objectives.triplet_hard_loss(t_fs5, i_feats.detach(), t_feats5.detach(), margin)
            '''
               
            loss = loss1 + loss2 + loss3 #+ loss4 + loss5
            ret.update({'mlm_loss': loss})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            #text_logits = self.classifier(t_feats1.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, image_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            #text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            #text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            #ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            
            #if epoch < 30:
            
            mlm_feats1, _ = self.base_model.encode_text(caption_ids1)
            x1 = self.cross_former(mlm_feats1, image_feats1, image_feats1)
            x1 = self.mlm_head(x1)  # [batch_size, text_len, num_colors]
            scores1 = x1.float().reshape(-1, self.args.vocab_size)
            mlm_labels1 = batch['mlm_labels1'].reshape(-1)
            pred1 = scores1.max(1)[1]
            mlm_label_idx1 = torch.nonzero(mlm_labels1)
            acc1 = (pred1[mlm_label_idx1] == mlm_labels1[mlm_label_idx1]).float().mean()
            loss1 = objectives.compute_mlm(scores1, mlm_labels1)
            '''
            mlm_feats2, _ = self.base_model.encode_text(caption_ids2)
            x2 = self.cross_former(mlm_feats2, image_feats2, image_feats2)
            x2 = self.mlm_head(x2)  # [batch_size, text_len, num_colors]
            scores2 = x2.float().reshape(-1, self.args.vocab_size)
            mlm_labels2 = batch['mlm_labels2'].reshape(-1)
            pred2 = scores2.max(1)[1]
            mlm_label_idx2 = torch.nonzero(mlm_labels2)
            acc2 = (pred2[mlm_label_idx2] == mlm_labels2[mlm_label_idx2]).float().mean()
            loss2 = objectives.compute_mlm(scores2, mlm_labels2)            
            

            mlm_feats3, _ = self.base_model.encode_text(caption_ids3)
            x3 = self.cross_former(mlm_feats3, image_feats3, image_feats3)
            x3 = self.mlm_head(x3)  # [batch_size, text_len, num_colors]
            scores3 = x3.float().reshape(-1, self.args.vocab_size)
            mlm_labels3 = batch['mlm_labels3'].reshape(-1)
            pred3 = scores3.max(1)[1]
            mlm_label_idx3 = torch.nonzero(mlm_labels3)
            acc3 = (pred3[mlm_label_idx3] == mlm_labels3[mlm_label_idx3]).float().mean()
            loss3 = objectives.compute_mlm(scores3, mlm_labels3)
            
            
            mlm_feats4, _ = self.base_model.encode_text(caption_ids4)
            x4 = self.cross_former(mlm_feats4, image_feats4, image_feats4)
            x4 = self.mlm_head(x4)  # [batch_size, text_len, num_colors]
            scores4 = x4.float().reshape(-1, self.args.vocab_size)
            mlm_labels4 = batch['mlm_labels4'].reshape(-1)
            pred4 = scores4.max(1)[1]
            mlm_label_idx4 = torch.nonzero(mlm_labels4)
            acc4 = (pred4[mlm_label_idx4] == mlm_labels4[mlm_label_idx4]).float().mean()
            loss4 = objectives.compute_mlm(scores4, mlm_labels4)
            
            mlm_feats5, _ = self.base_model.encode_text(caption_ids5)
            x5 = self.cross_former(mlm_feats5, image_feats5, image_feats5)
            x5 = self.mlm_head(x5)  # [batch_size, text_len, num_colors]
            scores5 = x5.float().reshape(-1, self.args.vocab_size)
            mlm_labels5 = batch['mlm_labels5'].reshape(-1)
            pred5 = scores5.max(1)[1]
            mlm_label_idx5 = torch.nonzero(mlm_labels5)
            acc5 = (pred5[mlm_label_idx5] == mlm_labels5[mlm_label_idx5]).float().mean()
            loss5 = objectives.compute_mlm(scores5, mlm_labels5)
            '''
            
            mlm_loss = loss1 #+ loss2 #+ loss3 #+ loss4 + loss5
            ret.update({'mlm_loss': mlm_loss})
            acc = acc1 #+ acc2 #+ acc3 #+ acc4 + acc5
            
            
            ret.update({'mlm_acc': acc})

        return ret

def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model