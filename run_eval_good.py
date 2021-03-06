"""Training script
This is the batch evaluation script for image denoising project.

Author: You-Yi Jau, Yiqian Wang
Date: 2020/03/30
"""

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
    """
    # class for manipulating sequences names
    """

    def __init__(self, dataset="kitti"):
        self.dataset = dataset
        pass

    @staticmethod
    def get_data_from_a_seq(seq):
        """
        # ['exp_name', 'model_deepF', 'model_SP', mode, 'new_eval_name',]
        """
        mode, exp_name, params = seq[2], seq[0], seq[1]
        filter = seq[3]
        filter_d = seq[4]
        new_eval_name = seq[5]
        return {
            "mode": mode,
            "exp_name": exp_name,
            "params": params,
            "filter": filter,
            "new_eval_name": new_eval_name,
            "filter_d": filter_d,
        }

    @staticmethod
    def update_config(config, mode=1, param=0, if_print=True, filter=None, filter_d=0):
        """
        # update config file for exporting
        """
        if mode == 0:
            pass
            # config['training']['pretrained'] = pretrained
            # config['training']['pretrained_SP'] = pretrained_SP
        elif mode == 1:
            config["data"]["augmentation"]["photometric"]["enable"] = True
            assert config["data"]["augmentation"]["photometric"]["enable"] == True
            config["data"]["augmentation"]["photometric"]["params"][
                "additive_gaussian_noise"
            ]["stddev_range"] = param
        elif mode == 2:
            config["data"]["augmentation"]["photometric"]["enable"] = True
            assert config["data"]["augmentation"]["photometric"]["enable"] == True
            config["data"]["augmentation"]["photometric"]["params"][
                "additive_gaussian_noise"
            ]["stddev_range"] = param
            config["model"]["filter"] = filter
            config["model"]["filter_d"] = filter_d

        if if_print and mode <= 5:
            logging.info(f"update params: {config['data']['augmentation']}")
        files_list = []

        return config, files_list

    @staticmethod
    def export_sequences(sequences, style="table", dataset="kitti", dump_name=None):
        export_seq = {}
        postfix = "k" if dataset == "kitti" else "a"
        for i, en in enumerate(sequences):
            export_seq[f"{en}.{postfix}"] = [
                f"{sequences[en][-1]}",
                "DeepF_err_ratio.npz",
            ] + sequences[en]
        print(f"sequences:")
        print(f"{sequences}")
        if dump_name is not None:
            file = f"configs/{dump_name}"
            with open(os.path.join(file), "w") as f:
                yaml.dump(export_seq, f, default_flow_style=True)
            logging.info(f"export sequences into {file}")

        pass

    @staticmethod
    def get_result_dict(seq_dict, base_path="", folder_idx=4, file_idx=""):
        files_dict = {}
        for i, en in enumerate(seq_dict):
            files_dict[en] = Path(base_path) / seq_dict[en][folder_idx] / Path(file_idx)

        return files_dict

    @staticmethod
    def load_print_file(file, if_print=False):
        allow_pickle = True
        exp = np.load(file, allow_pickle=allow_pickle)
        print(list(exp))
        # for i, en in enumerate(exp):
        #     print(f"exp[{en}], {exp[en]}")

        ## homography [1,3,5]
        result_list = []
        print(f"homo-1, homo-3, homo5, rep, MLE, NN mAP, mscores")
        homo = exp["correctness"][:, :3].mean(axis=0)
        result_list.extend(homo)
        # for h in homo:
        #     print(h)

        result_en = ["repeatability", "localization_err", "mAP", "mscore"]
        for i, en in enumerate(result_en):
            num = exp[en].mean()
            result_list.append(num)
        #     print(num)
        output = ", ".join([f"{r:.3}" for r in result_list])
        print(output)
        return output

    def get_sequences(self, name="", date="1107"):
        """
        # set sequence names and parameters
        sequences: 
        # ['exp_name', 'param', mode, 'filter', filter_d]
        """
        # gen_eval_name = lambda exp_name, iter, date: f"{exp_name}{iter}_{self.dataset}Testall_{date}"
        gen_eval_name = (
            lambda exp_name, fil, iter, date: f"{exp_name}{iter}_{fil}_{self.dataset}Test_{date}"
        )

        sift_sigma = {
            ## add filters
            ## sigma = 0
            # 'sift_sig-0-0': ['sift_sig_', [0.0,0.0], 2, 'None', 0],
            # 'sift_sig-0-1': ['sift_sig_', [0.0,0.0], 2, 'median', 3], # median
            # 'sift_sig-0-2': ['sift_sig_', [0.0,0.0], 2, 'bilateral'], # bilateral
            # 'sift_sig-0-3': ['sift_sig_', [0.0,0.0], 2, 'm_bilateral'], # m_bilateral
            # ## sigma = 5
            # 'sift_sig-5-0': ['sift_sig_', [5.0,5.0], 2, 'None', 0],
            # 'sift_sig-5-1': ['sift_sig_', [5.0,5.0], 2, 'median', 3], # 'bilateral
            # 'sift_sig-5-2': ['sift_sig_', [5.0,5.0], 2, 'bilateral'], # bilateral
            # 'sift_sig-5-3': ['sift_sig_', [5.0,5.0], 2, 'm_bilateral'], # bilateral
            # 'sift_sig-5-1-5': ['sift_sig_', [5.0,5.0], 2, 'median', 5], # 'bilateral
            # 'sift_sig-5-4': ['sift_sig_', [5.0,5.0], 2, 'm_bilateral_thd', 11], # bilateral_thd
            # 'sift_sig-5-5': ['sift_sig_', [5.0,5.0], 2, 'm_guided_thd', 1], # bilateral_thd
            ## sigma = 10
            # 'sift_sig-10-0': ['sift_sig_', [10.0,10.0], 2, 'None', 0],
            ###'sift_sig-10-1-7': ['sift_sig_', [10.0,10.0], 2, 'median', 7], # 'bilateral
            ###'sift_sig-10-1-9': ['sift_sig_', [10.0,10.0], 2, 'median', 9], # 'bilateral
            # 'sift_sig-10-2': ['sift_sig_', [10.0,10.0], 2, 'bilateral', 11], # bilateral
            "sift_sig-10-3": [
                "sift_sig_",
                [10.0, 10.0],
                2,
                "m_bilateral",
                11,
            ],  # bilateral
            # 'sift_sig-10-1-5': ['sift_sig_', [10.0,10.0], 2, 'median', 5], # 'bilateral
            # 'sift_sig-10-4': ['sift_sig_', [10.0,10.0], 2, 'm_bilateral_thd', 11], # bilateral
            # 'sift_sig-10-5': ['sift_sig_', [10.0,10.0], 2, 'm_guided_thd', 1], # bilateral
            # ## sigma = 15
            # 'sift_sig-15-0': ['sift_sig_', [15.0,15.0], 2, 'None', 0],
            # 'sift_sig-15-1': ['sift_sig_', [15.0,15.0], 2, 'median', 3], # 'bilateral
            # 'sift_sig-15-2': ['sift_sig_', [15.0,15.0], 2, 'bilateral'], # bilateral
            # 'sift_sig-15-3': ['sift_sig_', [15.0,15.0], 2, 'm_bilateral'], # bilateral
            # 'sift_sig-15-1-5': ['sift_sig_', [15.0,15.0], 2, 'median', 5], # 'bilateral
            # 'sift_sig-15-4': ['sift_sig_', [15.0,15.0], 2, 'm_bilateral_thd', 11], # bilateral_thd
            # 'sift_sig-15-5': ['sift_sig_', [15.0,15.0], 2, 'm_guided_thd', 1], # bilateral_thd
            # ## sigma = 20
            # 'sift_sig-20-0': ['sift_sig_', [20.0,20.0], 2, 'None', 0],
            # 'sift_sig-20-1': ['sift_sig_', [20.0,20.0], 2, 'median', 3], # 'bilateral
            # 'sift_sig-20-2': ['sift_sig_', [20.0,20.0], 2, 'bilateral'], # bilateral
            # 'sift_sig-20-3': ['sift_sig_', [20.0,20.0], 2, 'm_bilateral'], # bilateral
            # 'sift_sig-20-1-5': ['sift_sig_', [20.0,20.0], 2, 'median', 5], # 'bilateral
            # 'sift_sig-20-4': ['sift_sig_', [20.0,20.0], 2, 'm_bilateral_thd', 11], # bilateral_thd
            # 'sift_sig-20-5': ['sift_sig_', [20.0,20.0], 2, 'm_guided_thd', 1], # bilateral_thd
            # ## sigma = 25
            # 'sift_sig-25-0': ['sift_sig_', [25.0,25.0], 2, 'None', 0],
            # 'sift_sig-25-1': ['sift_sig_', [25.0,25.0], 2, 'median', 3], # 'bilateral
            # 'sift_sig-25-2': ['sift_sig_', [25.0,25.0], 2, 'bilateral'], # bilateral
            # 'sift_sig-25-3': ['sift_sig_', [25.0,25.0], 2, 'm_bilateral'], # bilateral
            # 'sift_sig-25-1-5': ['sift_sig_', [25.0,25.0], 2, 'median', 5], # 'bilateral
            # 'sift_sig-25-4': ['sift_sig_', [25.0,25.0], 2, 'm_bilateral_thd', 11], # bilateral_thd
            # 'sift_sig-25-5': ['sift_sig_', [25.0,25.0], 2, 'm_guided_thd', 1], # bilateral_thd
        }

        # corr_ablation = {
        #     ##### superpoint correspondences
        #     'Sp-k': ['superpoint_kitti_heat2_0', 50000, 50000, 6],
        #     'Sp-D-end-k-f': ['baselineTrain_kittiSp_deepF_end_kittiFLoss_freezeSp_v1', 45000, 45000, 6],
        #     'Sp-D-end-k-p': ['baselineTrain_kittiSp_kittiDeepF_end_kittiPoseLoss_v0', 8000, 8000, 6],
        #     'Sp-D-end-k-f-p': ['baselineTrain_kittiSp_deepF_end_kittiFLossPoseLoss_v1_freezeSp', 38000, 38000, 6],
        #     # 'Sp-a': ['superpoint_apollo_v1', 40000, 40000, 6],
        # }

        all_sequences = {"sift_sigma": sift_sigma}
        sequence = all_sequences.get(name, None)
        if sequence is None:
            logging.error(f"sequence name: {name} doesn't exist")
        else:
            idx_exp_name = 0
            idx_fil = 3
            idx_iter = 1
            for i, en in enumerate(sequence):
                eval_name = gen_eval_name(
                    sequence[en][idx_exp_name],
                    sequence[en][idx_iter][0],
                    f"{sequence[en][idx_fil]}{sequence[en][idx_fil+1]}",
                    date,
                )
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
    parser.add_argument(
        "exper_name",
        type=str,
        default="test",
        help="Experiment name and dates. Ex: 2sample_test0330",
    )
    parser.add_argument(
        "-m",
        "--model_base",
        type=str,
        default="sift",
        help="[sift], select the dataset your about to eval",
    )
    parser.add_argument(
        "-e",
        "--exper_path",
        type=str,
        default="./logs",
        help="the folder to logs and other checkpoint folders.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="hpatches",
        choices=["hpatches"],
        help="select the dataset your about to eval",
    )
    parser.add_argument(
        "-s",
        "--scp",
        type=str,
        default=None,
        help="[cephfs, KITTI, APOLLO, local], scp checkpoints from/to your current position",
    )
    parser.add_argument(
        "-sf",
        "--scp_from_server",
        action="store_true",
        help="send data from server to here",
    )

    parser.add_argument(
        "-ce",
        "--check_exist",
        action="store_true",
        help="check if exported sequences exist",
    )
    parser.add_argument(
        "-co",
        "--check_output",
        action="store_true",
        help="check if already ran the sequence",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="plot images in evaluation"
    )
    parser.add_argument(
        "--runEval", action="store_true", help="run export and evaluation"
    )
    parser.add_argument(
        "--runCorr",
        action="store_true",
        help="Not impplemented: run correspondences evaluation",
    )
    parser.add_argument(
        "-es",
        "--export_sequences",
        type=str,
        default=None,
        help="The name of dumped yaml",
    )

    args = parser.parse_args()
    print(args)
    dataset = args.dataset

    assert dataset == "hpatches", "your dataset is not supported"
    if_scp = True if args.scp is not None else False  ## not implemented

    scp_location = args.scp
    if_runEval = args.runEval
    if_runCorr = args.runCorr
    if_check_exist = args.check_exist
    if_check_output = args.check_output
    if_plot_img = args.plot
    exp_path = args.exper_path
    model_base = args.model_base
    scp_from_server = args.scp_from_server
    # load base config
    base_config = f"configs/classical_descriptors.yaml"

    with open(base_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # assert some settings
    assert config["training"]["reproduce"] == True, "reproduce should be 'true'"
    # assert config['training']['val_batches'] == -1 or config['training']['val_interval'] == -1, "val_batches should be -1 to use all test sequences"

    # load evaluation sequences
    seq_manager = sequence_info(dataset=dataset)
    sequence_dict = seq_manager.get_sequences(
        name=f"{model_base}_sigma", date=args.exper_name
    )  # Gamma1.5_1114
    logging.info(f"get sequence_dict: {sequence_dict}")

    if args.export_sequences is not None:
        seq_manager.export_sequences(
            sequence_dict, dataset=dataset, dump_name=args.export_sequences
        )

    def check_exit(file, entry="", should_exist=True):
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

        seq = sequence_dict
        data = seq_manager.get_result_dict(
            seq, base_path="./logs/", folder_idx=5, file_idx="./predictions/result.npz"
        )
        print(f"{data}")
        # check output one-by-one
        for i, en in enumerate(sequence_dict):
            if check_exit(f"{data[en]}", entry=en, should_exist=True):
                seq_manager.load_print_file(data[en])
        logging.info(f"++++++++ end check_output ++++++++")
        ## output to table

    if if_scp or if_check_exist:
        logging.info(f"++++++++ if_scp or if_check_exist ++++++++")
        for i, en in enumerate(sequence_dict):
            seq = sequence_dict[en]
            data = seq_manager.get_data_from_a_seq(seq)
            mode, exp_name, params = data["mode"], data["exp_name"], data["params"]
            filter = data.get("filter", None)
            filter_d = data.get("filter_d", 11)
            new_eval_name = data["new_eval_name"]
            # get config    
            temp_config, files = seq_manager.update_config(
                config,
                mode=mode,
                param=params,
                if_print=True,
                filter=filter,
                filter_d=filter_d,
            )
            # mkdir
            exp_dir = Path(f"{exp_path}/{exp_name}")
            exp_dir_checkpoint = exp_dir / "checkpoints"
            exp_dir_checkpoint.mkdir(parents=True, exist_ok=True)
            if if_check_output:
                new_eval_name = data["new_eval_name"]
                # check_exit(f"{exp_path}/{new_eval_name}/DeepF_err_ratio.npz", entry=en, should_exist=True)
                files = [f"{exp_path}/{new_eval_name}"]
            # else:
            #     files = [f'{exp_dir}/config.yml',
            #         temp_config['training']['pretrained'],
            #         temp_config['training']['pretrained_SP']]
            # for file in files:
            file = f"logs/{new_eval_name}"
            exist = Path(file).exists()
            if if_check_exist:
                if exist:
                    logging.info(f"{en}: {file} exist? {exist}")
                else:
                    logging.warning(f"{en}: {file} exist? {exist}")

        logging.info(f"++++++++ end if_scp or if_check_exist ++++++++")
        pass

    # if_runCorr = True
    if if_runEval or if_runCorr:
        # run sequences
        for i, en in enumerate(sequence_dict):
            seq = sequence_dict[en]
            data = seq_manager.get_data_from_a_seq(seq)
            # mode, exp_name, pretrained, pretrained_SP = data['mode'], data['exp_name'], data['pretrained'], data['pretrained_SP']
            mode, exp_name, params = data["mode"], data["exp_name"], data["params"]
            filter = data.get("filter", None)
            filter_d = data.get("filter_d", 11)
            new_eval_name = data["new_eval_name"]
            # update config
            temp_config, _ = seq_manager.update_config(
                config,
                mode=mode,
                param=params,
                if_print=True,
                filter=filter,
                filter_d=filter_d,
            )
            logging.info(f"temp_config: {temp_config}")
            temp_config_file = "temp_config_apo.yaml"
            # dump config
            with open(os.path.join("configs", temp_config_file), "w") as f:
                yaml.dump(temp_config, f, default_flow_style=False)
            if if_runEval and check_exit(f"{exp_path}/{new_eval_name}"):
                logging.error(f"{exp_path}/{new_eval_name} should not exist. Stopped!")
            commands = []
            # get commands
            if if_runEval:
                commands.append(
                    f"python export_classical.py export_descriptor configs/{temp_config_file} \
                        {new_eval_name}"
                )
                # logging.info(f"running command: {command}")
                # subprocess.run(f"{command}", shell=True, check=True)
                command_plot = (
                    "--outputImg --plotMatching" if if_plot_img == True else ""
                )
                commands.append(
                    f"python evaluation.py ./logs/{new_eval_name}/predictions \
                    --repeatibility --homography {command_plot}"
                )
            # run commands
            for command in commands:
                logging.info(f"running command: {command}")
                subprocess.run(f"{command}", shell=True, check=True)

