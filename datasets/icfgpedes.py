import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset
import pdb

class ICFGPEDES(BaseDataset):
    """
    ICFG-PEDES

    Reference:
    Semantically Self-Aligned Network for Text-to-Image Part-aware Person Re-identification arXiv 2107

    URL: http://arxiv.org/abs/2107.12666

    Dataset statistics:
    # identities: 4102
    # images: 34674 (train) + 4855 (query) + 14993 (gallery)
    # cameras: 15
    """
    dataset_dir = 'ICFG-PEDES'

    def __init__(self, args, root='', verbose=True):
        super(ICFGPEDES, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')
        self.n = args.ways
        self.k = args.shot
        name = 'one-shot-semi.json'

        self.anno_path = op.join(self.dataset_dir, name)
        self._check_before_run()
        self.unlab_id_path = []
        self.arg = args

        self.train_annos, self.test_annos, self.un_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)
        self.unlabel, self.unlab_id_container = self._process_anno_un(self.un_annos)

        if verbose:
            self.logger.info("=> ICFG-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, un_annos, val_annos = [], [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            elif anno['split'] == 'unlabel':
                un_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, un_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _process_anno_un(self, annos: List[dict], training=False):
        dataset = {}
        img_paths = []
        img_pids = []
        pid_container = set()
        image_id = 0
        for anno in annos:
            pid_container.add(image_id)
            img_pids.append(image_id)
            img_path = op.join(self.img_dir, anno['file_path'])
            img_paths.append(img_path)
            self.unlab_id_path.append((image_id, img_paths))
            image_id += 1
        dataset = {
            "image_pids": img_pids,
            "img_paths": img_paths
        }
        return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
