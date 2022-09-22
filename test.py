"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data

import torch
import numpy as np

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
#model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        
        input2 = (data_i['label'][b].float().squeeze(0)/2)-1

        real_image_t1 = data_i['image'][b][0,:,:]
        real_image_t1ce = data_i['image'][b][1,:,:]
        real_image_t2 = data_i['image'][b][2,:,:]
        real_image_flair = data_i['image'][b][3,:,:]

        real_image_all = torch.cat((input2, real_image_t1, real_image_t1ce, real_image_t2, real_image_flair),1)

        synthesized_image_t1 = generated[b][0,:,:]
        synthesized_image_t1ce = generated[b][1,:,:]
        synthesized_image_t2 = generated[b][2,:,:]
        synthesized_image_flair = generated[b][3,:,:]

        synthesized_image_all = torch.cat((input2, synthesized_image_t1, synthesized_image_t1ce, synthesized_image_t2, synthesized_image_flair),1)

        all = torch.cat((real_image_all, synthesized_image_all),0)

        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image_t1', synthesized_image_t1),
                               ('synthesized_image_t1ce', synthesized_image_t1ce),
                               ('synthesized_image_t2', synthesized_image_t2),
                               ('synthesized_image_flair', synthesized_image_flair),
                               ('all', all)
                              ])

        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
