# BORDE
BORDE: Boundary and Sub-Region Denormalization for Semantic Brain Image Synthesis

Paper: https://ieeexplore.ieee.org/document/9643093

ISLES pre-proccesed dataset: https://drive.google.com/file/d/1PO69crGf5KpmYJsCO3uz54y5X-nX8N2a/view?usp=sharing
BRATS pre-proccesed dataset: https://drive.google.com/file/d/1ai3r_rX0eex14qUJexHh2tvF9HMcm3QZ/view?usp=sharing

command for borde: python train.py --name brats19 --dataset_mode custom --label_dir datasets/brats2019/TRAIN/sem --image_dir datasets/brats2019/TRAIN/img --label_nc 5 --batchSize 4 --preprocess_mode_label gray --preprocess_mode_image four --load_size 256 --output_nc 4 --no_instance --ngf 32 --no_vgg_loss --netG borde

command for spade: python train.py --name brats19 --dataset_mode custom --label_dir datasets/brats2019/TRAIN/sem --image_dir datasets/brats2019/TRAIN/img --label_nc 5 --batchSize 4 --preprocess_mode_label gray --preprocess_mode_image four --load_size 256 --output_nc 4 --no_instance --ngf 32 --no_vgg_loss --netG spade

//command sean: python train.py --name brats19 --dataset_mode custom --label_dir datasets/brats2019/TRAIN/sem --image_dir datasets/brats2019/TRAIN/img --label_nc 5 --batchSize 4 --preprocess_mode_label gray --preprocess_mode_image four --load_size 256 --output_nc 4 --no_instance --ngf 32 --no_vgg_loss --netG sean

//command clade: python train.py --name brats19 --dataset_mode custom --label_dir datasets/brats2019/TRAIN/sem --image_dir datasets/brats2019/TRAIN/img --label_nc 5 --batchSize 4 --preprocess_mode_label gray --preprocess_mode_image four --load_size 256 --output_nc 4 --no_instance --ngf 32 --no_vgg_loss --netG clade
