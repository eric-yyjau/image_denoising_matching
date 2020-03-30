# ECE251C project: image_denoising_matching
https://docs.google.com/document/d/1VCM1yOlSXhzatvEgNLB1IoWqT81NWjPtbGr0THJ5uqE/edit#heading=h.nrpj9v3j7ji7

## To do
- Study and implement the bilateral filter, multiresolution bilateral filter, guided filter, DnCNN and DUDnCNN.
- Compare the image denoising and edge-preserving performance of the above algorithms, by peak signal to noise ratio (PSNR) and structural similarity (SSIM) index. (COCO dataset)
- Further compare the feature preserving performance by comparing the feature detection and matching result before and after denoising using SIFT. The metric is repeatability and homography estimation on COCO dataset. (or Hpatches[9] benchmark dataset)


### schedule
- 11/25 progress report
- 12/9 final presentation
- 12/14 final report

## Run the code


### 2) Export and Evaluate repeatability on SIFT
#### Requirements
- GPU
  - need a gpu to run (not work on pure cpu version)

#### Datasets
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
    - Please put the dataset under `datasets/`.

#### Export
- set `config` file.
```
python export_classical.py export_descriptor configs/example_config.yaml sift_test_small
```
#### evaluate
```
python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]
python evaluation.py logs/sift_test_small/predictions -r -homo
```
<!-- - specify the pretrained model -->

#### Run evaluation for different noise
```
# check help 
python run_eval_good.py -h
```
- Change the parameters from `sequence_info.get_sequences`.
  - set `['exp_name', 'param', mode, 'filter', filter_d]`.
- Run for collecting samples
```
python run_eval_good.py test_0330 --dataset hpatches --model sift --runEval
```
- Check output and check exist
```
python run_eval_good.py test_0330 --dataset hpatches --model sift -co -ce
```



#### Results



