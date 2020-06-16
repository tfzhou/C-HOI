import os
import torch

pretrained_file = '/home/ubuntu/.cache/torch/checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'

num_class = 144

pretrained_weights = torch.load(pretrained_file)
pretrained_weights['state_dict']['bbox_head.0.fc_cls.weight'].resize_(num_class, 1024)
pretrained_weights['state_dict']['bbox_head.0.fc_cls.bias'].resize_(num_class)
pretrained_weights['state_dict']['bbox_head.1.fc_cls.weight'].resize_(num_class, 1024)
pretrained_weights['state_dict']['bbox_head.1.fc_cls.bias'].resize_(num_class)
pretrained_weights['state_dict']['bbox_head.2.fc_cls.weight'].resize_(num_class, 1024)
pretrained_weights['state_dict']['bbox_head.2.fc_cls.bias'].resize_(num_class)

pretrained_weights['state_dict']['mask_head.0.conv_logits.weight'].resize_(num_class, 256, 1, 1)
pretrained_weights['state_dict']['mask_head.0.conv_logits.bias'].resize_(num_class)
pretrained_weights['state_dict']['mask_head.1.conv_logits.weight'].resize_(num_class, 256, 1, 1)
pretrained_weights['state_dict']['mask_head.1.conv_logits.bias'].resize_(num_class)
pretrained_weights['state_dict']['mask_head.2.conv_logits.weight'].resize_(num_class, 256, 1, 1)
pretrained_weights['state_dict']['mask_head.2.conv_logits.bias'].resize_(num_class)

torch.save(pretrained_weights, "{}_{}.pth".format(pretrained_file[:-4], num_class))
