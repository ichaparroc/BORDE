"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)


def plot_images(dataset): #-#
    datas = []
    datas = next(iter(dataset))
    #for i, data in datas:
    for i, data_i in enumerate(dataloader):
        print(data_i['label'].size())
        print(data_i.keys())
        print(np.unique(data_i['label'].numpy()))
        plt.figure(figsize=(2, 2), dpi = data_i['label'].shape[2])
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(data_i['label'][:2*2,:,:,:], nrow = 2, normalize = True),(1,2,0)))
        plt.show()
        input("press intro")

        print(data_i['image'].size())
        print(data_i.keys())
        print(np.unique(data_i['image'].numpy()))

        print("this:",data_i['image'][:2*2,0,:,:].size())
        
        plt.figure(figsize=(2, 2), dpi = data_i['image'].shape[2])
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(data_i['image'][:2*2,0,:,:].unsqueeze(1), nrow = 2, normalize = True),(1,2,0)))
        plt.show()

        plt.figure(figsize=(2, 2), dpi = data_i['image'].shape[2])
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(data_i['image'][:2*2,1,:,:].unsqueeze(1), nrow = 2, normalize = True),(1,2,0)))
        plt.show()

        plt.figure(figsize=(2, 2), dpi = data_i['image'].shape[2])
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(data_i['image'][:2*2,2,:,:].unsqueeze(1), nrow = 2, normalize = True),(1,2,0)))
        plt.show()

        plt.figure(figsize=(2, 2), dpi = data_i['image'].shape[2])
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(data_i['image'][:2*2,3,:,:].unsqueeze(1), nrow = 2, normalize = True),(1,2,0)))
        plt.show()
        input("press intro")
        
        break;

#Sample images
#plot_images(dataloader)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            synthesized_image = trainer.get_latest_generated()
            
            synthesized_image_t1 = synthesized_image[:,0,:,:].unsqueeze(1)
            synthesized_image_t1ce = synthesized_image[:,1,:,:].unsqueeze(1)
            synthesized_image_t2 = synthesized_image[:,2,:,:].unsqueeze(1)
            synthesized_image_flair = synthesized_image[:,3,:,:].unsqueeze(1)

            real_image_t1 = data_i['image'][:,0,:,:].unsqueeze(1)
            real_image_t1ce = data_i['image'][:,1,:,:].unsqueeze(1)
            real_image_t2 = data_i['image'][:,2,:,:].unsqueeze(1)
            real_image_flair = data_i['image'][:,3,:,:].unsqueeze(1)



            #visuals = OrderedDict([('input_label', data_i['label']),
            #                       ('synthesized_image', trainer.get_latest_generated()),
            #                       ('real_image', data_i['image'])])
            
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image_t1', synthesized_image_t1),
                                   ('synthesized_image_t1ce', synthesized_image_t1ce),
                                   ('synthesized_image_t2', synthesized_image_t2),
                                   ('synthesized_image_flair', synthesized_image_flair),
                                   ('real_image_t1', real_image_t1),
                                   ('real_image_t1ce', real_image_t1ce),
                                   ('real_image_t2', real_image_t2),
                                   ('real_image_flair', real_image_flair)
                                   #('synthesized_image_all', synthesized_image_all)
                                  ])

            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
