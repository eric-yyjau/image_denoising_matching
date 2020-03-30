# Feature-preserving image denoising with multiresolution filters
Final project for ECE251C: Filter banks and wavelets

## Summary
- Study and implement the bilateral filter, multiresolution bilateral filter, guided filter.
- Compare the image denoising and edge-preserving performance of the above algorithms, by peak signal to noise ratio (PSNR) and structural similarity (SSIM) index. (SIDD dataset)
- Further compare the feature preserving performance by comparing the feature detection and matching result before and after denoising using SIFT. The metric is repeatability and homography estimation on Hpatches benchmark dataset.


## Run the code
### 1) Denoising performance on SIDD
- download SIDD small sRGB from ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Small_sRGB_Only.zip
- run 'demo_filters.py' to see result on one patch
```
python demo_filters.py # run on an example image
```
- run 'tune_filters_sidd.py' to tune parameters on 40 images
- run 'test_filters_sidd.py' to test filters on SIDD


### 2) Export and Evaluate repeatability on SIFT
#### Requirements
- GPU
  - need a gpu to run (not work on pure cpu version)
- Environments
  - python 3.6, pytorch >= 0.4.1
  - conda create
```
conda env create -f environment.yml # env name: py36-imgdn
```
  - use pip 
```
conda create --name py36-imgdn python=3.6
conda activate py36-imgdn
pip install -r requirements.txt
```

#### Datasets
- download HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
    - Please put the dataset under `datasets/`.
```
`-- datasets/
|   |-- HPatches
|   |   |-- i_ajuntament
|   |   | ...
`-- ...
```

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
#### Run scripts
- Do export, then evaluate the prediction.
```
./run_export.sh
```

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
Please refer to `final_report.pdf` and `presentation.pdf`.
- sift results
  - matching
![Matching](./data/sift_results/0m.png)
  - repeatability
![Matching](./data/sift_results/0.png)
  - warping using predicted homography
![Matching](./data/sift_results/0_warp.png)

## Logging
https://docs.google.com/document/d/1VCM1yOlSXhzatvEgNLB1IoWqT81NWjPtbGr0THJ5uqE/edit#heading=h.nrpj9v3j7ji7

# Credits
This implementation is developed by [Yigian Wang](https://github.com/yiqian-wang) and [You-Yi Jau](https://github.com/eric-yyjau).
