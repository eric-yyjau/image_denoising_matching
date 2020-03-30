# run to do exporting and evaluation
export_folder='sift_test_small'
echo $export_folder
python export_classical.py export_descriptor configs/example_config.yaml $export_folder
python3 evaluation.py ./logs/$export_folder/predictions --repeatibility --outputImg --homography --plotMatching
