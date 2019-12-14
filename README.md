# ECE251C project: image_denoising_matching
https://docs.google.com/document/d/1VCM1yOlSXhzatvEgNLB1IoWqT81NWjPtbGr0THJ5uqE/edit#heading=h.nrpj9v3j7ji7

## To do
- Study and implement the bilateral filter, multiresolution bilateral filter, guided filter.
- Compare the image denoising and edge-preserving performance of the above algorithms, by peak signal to noise ratio (PSNR) and structural similarity (SSIM) index. (SIDD dataset)
- Further compare the feature preserving performance by comparing the feature detection and matching result before and after denoising using SIFT. The metric is repeatability and homography estimation on COCO dataset. (or Hpatches[9] benchmark dataset)


### schedule
- 11/25 progress report
- 12/9 final presentation
- 12/14 final report

## Run the code
### 1) Denoising performance on SIDD
- download SIDD small sRGB from ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Small_sRGB_Only.zip
- run 'demo_filters.py' to see result on one patch
- run 'tune_filters_sidd.py' to tune parameters on 40 images
- run 'test_filters_sidd.py' to test filters on SIDD


### 2) Export and Evaluate repeatability on SIFT
#### Export
```
python export_classical.py export_descriptor configs/classical_descriptors.yaml sift_test
```
#### evaluate
```
python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]
python evaluation.py logs/sift_test/predictions -r -o
```
- specify the pretrained model

### Run tensorboard
```
tensorboard --logdir=runs/[ train_base | train_joint ]
```

### Current best model
```
logs/superpoint_coco30_1/checkpoints/superPointNet_170000_checkpoint.pth.tar
```

