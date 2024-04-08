

def Compact_Matching(image_fetures, text_fetures, identity_labels, logit_scale, epsilon=1e-8):

    batch_size = image_fetures.shape[0]
    identity_labels = identity_labels.reshape((batch_size, 1)) # make sure identity_labels size is [batch_size, 1]
    label_dist = identity_labels - identity_labels.t()
    labels = (label_dist == 0).float()

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    sim_matrix = F.softmax(logit_scale*t2i_cosine_theta, dim=1)*F.softmax(logit_scale*t2i_cosine_theta, dim=0)

    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_t =F.cross_entropy(sim_matrix.t(), labels)
    loss = (loss_i +  loss_t)/2

    return loss, sim_matrix, t2i_cosine_theta


def hard_loss(t2i_cosine_theta1, t2i_cosine_theta2, t2i_cosine_theta3, identity_labels, logit_scale):
    
    batch_size = image_fetures.shape[0]
    identity_labels = identity_labels.reshape((batch_size, 1)) # make sure identity_labels size is [batch_size, 1]
    label_dist = identity_labels - identity_labels.t()
    labels = (label_dist == 0).float()

    pos = torch.min(torch.min(t2i_cosine_theta1, t2i_cosine_theta2), t2i_cosine_theta3)
    neg = torch.max(torch.max(t2i_cosine_theta1, t2i_cosine_theta2), t2i_cosine_theta3)
   
    t2i_cosine_theta = labels*pos + (1-labels)*neg
    sim_matrix = F.softmax(logit_scale*t2i_cosine_theta, dim=1)*F.softmax(logit_scale*t2i_cosine_theta, dim=0)

    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_t =F.cross_entropy(sim_matrix.t(), labels)
    loss = (loss_i +  loss_t)/2 
   
    return loss
