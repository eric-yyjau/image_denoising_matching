# extract tars

import subprocess
import glob
import yaml
import logging
import os
from pathlib import Path
import argparse
from utils.logging import *
import numpy as np

class sequence_info(object):
    def __init__(self, dataset='kitti'):
        self.dataset = dataset
        pass

    @staticmethod
    def get_data_from_a_seq(seq):
        """
        # ['exp_name', 'model_deepF', 'model_SP', mode, 'new_eval_name',]
        """
        mode, exp_name, params = seq[2], seq[0], seq[1]
        new_eval_name = seq[3]
        return {
            'mode': mode, 'exp_name': exp_name, 'params': params,
            'new_eval_name': new_eval_name,
        }

    @staticmethod
    def update_config(config, mode=1, param=0, if_print=True):
        if mode == 0:
            pass
            # config['training']['pretrained'] = pretrained
            # config['training']['pretrained_SP'] = pretrained_SP
        elif mode == 1:
            config['data']['augmentation']['photometric']['enable'] = True
            assert config['data']['augmentation']['photometric']['enable'] == True
            config['data']['augmentation']['photometric']['params']['additive_gaussian_noise']['stddev_range'] = param
        elif mode == 2:
            config['data']['augmentation']['photometric']['enable'] = True
            assert config['data']['augmentation']['photometric']['enable'] == True
            config['data']['augmentation']['photometric']['params']['additive_gaussian_noise']['stddev_range'] = param
            config['model']['filter'] = 'bilateral'

        if if_print and mode <= 5:
            logging.info(f"update params: {config['data']['augmentation']}")
        files_list = []

        return config, files_list

    @staticmethod
    def export_sequences(sequences, style='table', dataset='kitti', dump_name=None):
        export_seq = {}
        postfix = 'k' if dataset == 'kitti' else 'a'
        for i, en in enumerate(sequences):
            export_seq[f"{en}.{postfix}"] = [f"{sequences[en][-1]}", 'DeepF_err_ratio.npz'] + sequences[en]
        print(f"sequences:")
        print(f"{sequences}")
        if dump_name is not None:
            file = f"configs/{dump_name}"
            with open(os.path.join(file), "w") as f:
                yaml.dump(export_seq, f, default_flow_style=True)
            logging.info(f"export sequences into {file}")


        pass

    def get_sequences(self, name='', date='1107'):
        """
        sequences: 
        # ['exp_name', 'param', mode, 'new_eval_name',]
        """
        # gen_eval_name = lambda exp_name, iter, date: f"{exp_name}{iter}_{self.dataset}Testall_{date}"
        gen_eval_name = lambda exp_name, iter, date: f"{exp_name}{iter}_{self.dataset}Test_{date}"

        sift_sigma = {
            # 'sift_sig-1': ['sift_sig_', [5.0,5.0], 1],
            # 'sift_sig-2': ['sift_sig_', [10.0,10.0], 1],

            ## add filters
            # 'sift_sig-11': ['bi_sift_sig_', [5.0,5.0], 2], # 'bilateral
            # 'sift_sig-12': ['bi_sift_sig_', [10.0,10.0], 2], # bilateral

            'sift_sig-3': ['sift_sig_', [15.0,15.0], 1],
            'sift_sig-13': ['sift_sig_', [15.0,15.0], 2],

            # 'sift_sig-4': ['sift_sig_', [20.0,20.0], 1],
            # 'sift_sig-5': ['sift_sig_', [25.0,25.0], 1],
            # 'sift_sig_1': ['sift_sig_v1_', [5.0,5.0], 1],
            # 'sift_sig_2': ['sift_sig_v2_', [5.0,5.0], 1],
        }

        # corr_ablation = {
        #     ##### superpoint correspondences
        #     'Sp-k': ['superpoint_kitti_heat2_0', 50000, 50000, 6],
        #     'Sp-D-end-k-f': ['baselineTrain_kittiSp_deepF_end_kittiFLoss_freezeSp_v1', 45000, 45000, 6],
        #     'Sp-D-end-k-p': ['baselineTrain_kittiSp_kittiDeepF_end_kittiPoseLoss_v0', 8000, 8000, 6],
        #     'Sp-D-end-k-f-p': ['baselineTrain_kittiSp_deepF_end_kittiFLossPoseLoss_v1_freezeSp', 38000, 38000, 6],
        #     # 'Sp-a': ['superpoint_apollo_v1', 40000, 40000, 6],
        # }

        all_sequences = {'sift_sigma': sift_sigma}
        sequence = all_sequences.get(name, None)
        if sequence is None:
            logging.error(f"sequence name: {name} doesn't exist")
        else:
            idx_exp_name = 0
            idx_iter = 1
            for i, en in enumerate(sequence):
                eval_name = gen_eval_name(sequence[en][idx_exp_name], sequence[en][idx_iter][0], date)
                sequence[en].extend([eval_name])
        return sequence
            # 1, 'superpoint_kitti_heat2_0', 50000,  ## no need to run

        pass



if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # level=logging.INFO,
        level=logging.INFO,
    )

    # add parser
    parser = argparse.ArgumentParser()
    # Training command
    parser.add_argument("exper_name", type=str, help='hpatches, select the dataset your about to eval')
    parser.add_argument("-m", "--model_base", type=str, default='sift', help='[sift], select the dataset your about to eval')
    parser.add_argument("-e", "--exper_path", type=str, default='./logs', help='the folder to logs and other checkpoint folders.')
    parser.add_argument("-s", "--scp", type=str, default=None, help="[cephfs, KITTI, APOLLO, local], scp checkpoints from/to your current position")
    parser.add_argument("-sf", "--scp_from_server", action="store_true", help="send data from server to here")
    
    parser.add_argument("-ce", "--check_exist", action="store_true", help="scp checkpoints to your current position")
    parser.add_argument("-co", "--check_output", action="store_true", help="check if already ran the sequence")
    parser.add_argument("--runEval", action="store_true", help="run evaluation")
    parser.add_argument("--runCorr", action="store_true", help="run correspondences evaluation")
    parser.add_argument("-es", "--export_sequences",  type=str, default=None, help="The name of dumped yaml")
    # parser.add_argument("server", type=str, default='theia', help='scp from theia or hyperion')
    args = parser.parse_args()
    print(args)
    dataset = args.exper_name
    # assert dataset == 'kitti' or dataset == 'apollo', 'your dataset is not supported'
    assert dataset == 'hpatches', 'your dataset is not supported'
    if_scp = True if args.scp is not None else False
    scp_location = args.scp
    if_runEval = args.runEval
    if_runCorr = args.runCorr
    if_check_exist = args.check_exist
    if_check_output = args.check_output
    exp_path = args.exper_path
    model_base = args.model_base
    scp_from_server = args.scp_from_server
    # load base config
    
    base_config = f'configs/classical_descriptors.yaml'


    with open(base_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # assert some settings
    assert config['training']['reproduce'] == True, "reproduce should be 'true'"
    # assert config['training']['val_batches'] == -1 or config['training']['val_interval'] == -1, "val_batches should be -1 to use all test sequences"


    # download
    seq_manager = sequence_info(dataset=dataset)
    sequence_dict = seq_manager.get_sequences(name=f'{model_base}_sigma', date='50sample_1120')  # Gamma1.5_1114
    logging.info(f"get sequence_dict: {sequence_dict}")

    if args.export_sequences is not None:
        seq_manager.export_sequences(sequence_dict, dataset=dataset, dump_name=args.export_sequences)

        # ['baselineEval_kittiSp_deepF_kittiPoseLoss_v1_16k_apolloTestall', 1, 'baselineTrain_kittiSp_deepF_kittiPoseLoss_v1', 16000, 16000],
        # ['baselineEval_kittiSp_deepF_end_kittiFLossPoseLoss_v1_freezeSp_38k_kittiTestall', 1, 'baselineTrain_kittiSp_deepF_end_kittiFLossPoseLoss_v1_freezeSp', 38000, 38000],
        # ['', 1, 'baselineTrain_apolloSp_deepF_apolloFLoss_v0', 24000, 24000],

    def check_exit(file, entry='', should_exist=True):
        exist = Path(file).exists()
        msg = f"{entry}: {file} exist? {exist}"
        if (exist and should_exist) or (not exist and not should_exist):
            logging.info(msg)
        else:
            logging.warning(msg)
        return exist
    
    if if_check_output:
        # check_files = "predictions/result.npz" if model_base == "sift" else "DeepF_err_ratio.npz" 
        check_files = "predictions/result.npz" if model_base == "sift" else ""
        logging.info(f"++++++++ check_output ++++++++")
        for i, en in enumerate(sequence_dict):
            seq = sequence_dict[en]        
            data = seq_manager.get_data_from_a_seq(seq)
            new_eval_name = data['new_eval_name']
            check_exit(f"{exp_path}/{new_eval_name}/{check_files}", entry=en, should_exist=True)       
        logging.info(f"++++++++ end check_output ++++++++")
        

    if if_scp or if_check_exist:
        logging.info(f"++++++++ if_scp or if_check_exist ++++++++")
        for i, en in enumerate(sequence_dict):
            seq = sequence_dict[en]
            data = seq_manager.get_data_from_a_seq(seq)
            mode, exp_name, pretrained, pretrained_SP = data['mode'], data['exp_name'], data['pretrained'], data['pretrained_SP']
            temp_config, files = seq_manager.update_config(config, mode, exp_path, exp_name, pretrained, pretrained_SP)
            # mkdir
            exp_dir = Path(f'{exp_path}/{exp_name}')
            exp_dir_checkpoint = exp_dir/'checkpoints'
            exp_dir_checkpoint.mkdir(parents=True, exist_ok=True)
            if if_check_output:
                new_eval_name = data['new_eval_name']
                # check_exit(f"{exp_path}/{new_eval_name}/DeepF_err_ratio.npz", entry=en, should_exist=True)
                files = [f"{exp_path}/{new_eval_name}"]
            # else:
            #     files = [f'{exp_dir}/config.yml', 
            #         temp_config['training']['pretrained'],
            #         temp_config['training']['pretrained_SP']]
            for file in files:
                exist = Path(file).exists()
                if if_check_exist:
                    if exist:
                        logging.info(f"{en}: {file} exist? {exist}")
                    else:
                        logging.warning(f"{en}: {file} exist? {exist}")
                if if_scp:
                    from_server = scp_from_server
                    if from_server:
                        assert exist == False, f'{file} already exists, stoped.'
                    else:
                        assert exist == True, f'{file} not exists to scp, stoped.'
                    # command = f"scp -r yyjau@hyperion.ucsd.edu:/home/yyjau/Documents/deepSfm/{file} {file}"
                    # command = f"scp -r yyjau@theia.ucsd.edu:/home/yyjau/Documents/deepSfm/{file} {file}"
                    # logging.info(f"[run command] {command}")
                    # subprocess.run(f"{command}", shell=True, check=True)
                    if len(file) > 0:
                        scp_to_server(file, end_name=scp_location, from_server=from_server)
        logging.info(f"++++++++ end if_scp or if_check_exist ++++++++")
        pass

    # if_runCorr = True
    if if_runEval or if_runCorr:
        # run sequences
        for i, en in enumerate(sequence_dict):
            seq = sequence_dict[en]
            data = seq_manager.get_data_from_a_seq(seq)
            # mode, exp_name, pretrained, pretrained_SP = data['mode'], data['exp_name'], data['pretrained'], data['pretrained_SP']
            mode, exp_name, params = data['mode'], data['exp_name'], data['params']
            new_eval_name = data['new_eval_name']
            # update config
            temp_config, _ = seq_manager.update_config(config, mode=mode, param=params, if_print=True)
            logging.info(f"temp_config: {temp_config}")
            temp_config_file = "temp_config_apo.yaml"
            # dump config
            with open(os.path.join("configs", temp_config_file), "w") as f:
                yaml.dump(temp_config, f, default_flow_style=False)
            if if_runEval and check_exit(f'{exp_path}/{new_eval_name}'):
                logging.error(f'{exp_path}/{new_eval_name} should not exist. Stopped!')
            commands = []
            if if_runEval:
                commands.append(f"python export_classical.py export_descriptor configs/{temp_config_file} \
                        {new_eval_name}")
                # logging.info(f"running command: {command}")
                # subprocess.run(f"{command}", shell=True, check=True)
                commands.append(f"python evaluation.py ./logs/{new_eval_name}/predictions \
                    --repeatibility --outputImg --homography --plotMatching")
            for command in commands:
                logging.info(f"running command: {command}")
                subprocess.run(f"{command}", shell=True, check=True)


    # python train_good_corr_4_vals_goodF_baseline.py eval_good configs/temp_config.yml \
    # baselineEval_kittiSp_kittideepF_notrain_kitti_testall_v0  --eval --test
