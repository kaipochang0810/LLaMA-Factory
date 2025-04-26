import os
import logging
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader
import torch
import json
import random
from copy import deepcopy
from utils.user_prompts import user_prompts
from utils.helper import device_print, device_0_breakpoint

logger = logging.getLogger(__name__)

class ValDataset(Dataset):

    def __init__(self, anno_file, dataset_name, config, **kwargs):
        self.dataset_name = dataset_name
        self.anno = json.load(open(anno_file, 'r'))
        self.entry_list = []
        for entry in self.anno:
            entry["video_fp"] = os.path.join(config.video_folder, entry["file_path"])
            if not os.path.exists(entry["video_fp"]):
                # device_print(f"video_fp: {entry['video_fp']} does not exist", config.device)
                continue
            ref_captions = {
                'dense': entry.pop('dense_caption'),
                'main_object': entry.pop('main_object_caption'),
                'background': entry.pop('background_caption')
            }
            all_caption_types = ['dense', 'main_object', 'background']
            for caption_type in all_caption_types:
                entry["caption_type"] = caption_type
                entry["ref_caption"] = ref_captions[caption_type]
                self.entry_list.append(entry.copy())

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        sources = self.entry_list[index]
        prompt = random.choice(user_prompts[sources["caption_type"]])
        ref_caption = sources["ref_caption"]
        clip_id = sources["clip_id"]
        video_fp = sources["video_fp"]
        caption_type = sources["caption_type"]


        return (clip_id, prompt, ref_caption, video_fp, caption_type)



def create_dataset(config):
    val_files = {}
    for val_name in config.val_tag.split('#'):
        if val_name not in config.val_file_dict:
            raise NotImplementedError(f"{val_name} not found in val_file_dict")
        val_files[val_name] = config.val_file_dict[val_name]

    subset_val_datasets = []
    for k, v in val_files.items():
        datasets = []
        if type(v[0]) != list:
            v = [v]
        for val_file in v:
            dataset = ValDataset(anno_file=val_file, dataset_name=k, config=config)
            total_size = len(dataset)
            chunk_size = total_size // config.num_chunks
            start_idx = config.chunk_idx * chunk_size
            end_idx = total_size if config.chunk_idx == config.num_chunks - 1 else (config.chunk_idx + 1) * chunk_size
            device_print(f"dataset: {k}, data_start_idx: {start_idx}, data_end_idx: {end_idx}", config.device)
            subset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
            datasets.append(subset)
        subset_val_datasets.append(ConcatDataset(datasets))

    return subset_val_datasets



def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)
    return loaders


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data


def collate_fn(batch):
    # prompts, ref_captions, scene_ids, bev_mf_paths, qids, obj_id, pred_ids, type_infos = zip(*batch)
    clip_ids, prompts, ref_captions, video_fps, caption_types = zip(*batch)
    return {
        "custom_prompt": prompts,
        "ref_caption": ref_captions,
        "clip_id": clip_ids,
        "video_fp": video_fps,
        "caption_type": caption_types
    }