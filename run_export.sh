
export_folder='sift_base_10_v0'
echo $export_folder
# python3 export2.py export_descriptor configs/hpatches_rep/magicpoint_repeatability_heatmap.yaml $export_folder
# python3 export2.py export_descriptor configs/magicpoint_repeatability_heatmap.yaml $export_folder
# python3 export.py export_descriptor_coco configs/magicpoint_repeatability.yaml $export_folder
# python export_classical.py export_descriptor configs/classical_descriptors.yaml $export_folder
python3 evaluation.py ./logs/$export_folder/predictions --repeatibility --outputImg --homography --plotMatching
