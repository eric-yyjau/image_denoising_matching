
class Exp_table_processor(Result_processor):
    """
    # process the results of different sequences.
    # sort into table
    """

    def __init__(self):
        self.folders = []
        pass


    def add_folders(self, folders):
        self.folders = folders
        pass

    @staticmethod
    def read_folder_list(folders, base_path="",  file_path=""):
        files_dict = {}
        for i, en in enumerate(seq_dict):
            pass
            

    @staticmethod
    def read_file_list(seq_dict, base_path="", folder_idx=0, file_idx=1):
        files_dict = {}
        for i, en in enumerate(seq_dict):
            files_dict[en] = (
                Path(base_path) / seq_dict[en][folder_idx] / seq_dict[en][file_idx]
            )
        return files_dict