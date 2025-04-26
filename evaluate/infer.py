import os
import json
import torch
from filelock import FileLock
from tqdm import tqdm
from val_dataset import create_dataset, create_loader, collate_fn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from utils.config_utils import setup_main
from utils.helper import device_print


def save_preds_to_file(processing_preds, predictions_file, device):
    remaining = []
    lock_path = predictions_file + ".lock"
    lock = FileLock(lock_path)
    # only one process at a time may open & append
    with lock:
        with open(predictions_file, "a", encoding="utf-8") as f:
            for proc_pred in processing_preds:
                if len(proc_pred.get("pred_captions", [])) == 3 and len(proc_pred.get("gt_captions", [])) == 3:
                    f.write(json.dumps(proc_pred, ensure_ascii=False) + "\n")
                else:
                    remaining.append(proc_pred)
    device_print(f"{len(processing_preds) - len(remaining)} samples are saved to {predictions_file}", device)
    return remaining


def inference(model, val_loader, device, config):
    eval_name = val_loader.dataset.datasets[0].dataset.dataset_name
    output_dir = os.path.join(config.output_dir, f"{eval_name}", "predictions", f"{model}")
    predictions_file = os.path.join(output_dir, f"pred_{config.iter_idx}.jsonl")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(predictions_file):
        with open(predictions_file, "r", encoding="utf-8") as f:
            finished_preds = [json.loads(line) for line in f]
        finished_clip_ids = {item["clip_id"] for item in finished_preds}
    else:
        open(predictions_file, "w", encoding="utf-8").close()
        finished_preds = []
        finished_clip_ids = set()
    device_print(f"Loaded {len(finished_preds)} existing predictions from {predictions_file}.", device)

    save_interval = config.save_interval
    batch_count = 0

    if model == 'qwenvl':
        device_print(f"Loading model from {config.model_path}", device)
        vmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        vmodel.eval()
        vmodel.to(device)
        vmodel.config.vision_config.temporal_size = 1  # to match process of llava-video
        vmodel.visual.patch_embed.temporal_patch_size = 1  # to match process of llava-video
        vmodel.visual.config.temporal_patch_size = 1  # to match process of llava-video
        
        processor = AutoProcessor.from_pretrained(config.model_path)
        processor.tokenizer.padding_side = "left"
        processor.image_processor.temporal_patch_size = 1  # to match process of llava-video

    processing_preds = []
    with tqdm(total=len(val_loader), desc=f"Inference {eval_name}") as pbar:
        for batch in val_loader:
            unfinished_indices = [
                i for i, clip_id in enumerate(batch["clip_id"]) if clip_id not in finished_clip_ids
            ]

            if not unfinished_indices:
                pbar.update(1)
                continue

            batch_size = len(unfinished_indices)
            unfinished_batch = {k: [v[i] for i in unfinished_indices] for k, v in batch.items()}

            for k in unfinished_batch.keys():
                if isinstance(unfinished_batch[k], torch.Tensor):
                    unfinished_batch[k] = unfinished_batch[k].to(device)

            if model == 'qwenvl':
                from qwen2_vl import get_response_qwen as get_response
                pred = get_response(vmodel, processor, unfinished_batch['custom_prompt'], unfinished_batch['video_fp'], config, eval_name)

            # save the preds for unfinished_indices into processing_preds
            for i, index in enumerate(unfinished_indices):
                # check if this clip_id is in the the processing, then only add the pred to pred_captions
                is_processing = False
                for proc_pred in processing_preds:
                    if batch["clip_id"][index] == proc_pred["clip_id"]:
                        is_processing = True
                        proc_pred["pred_captions"].update({f"{batch['caption_type'][index]}": pred[i]})
                        proc_pred["gt_captions"].update({f"{batch['caption_type'][index]}": batch['ref_caption'][index]})
                        break
                if not is_processing:
                    pred_captions = {f"{batch['caption_type'][index]}": pred[i]}
                    gt_captions = {f"{batch['caption_type'][index]}": batch['ref_caption'][index]}
                    processing_preds.append({
                        "clip_id": batch["clip_id"][index],
                        "prompt": batch["custom_prompt"][index],
                        "pred_captions": pred_captions,
                        "gt_captions": gt_captions,
                    })

            batch_count += 1
            pbar.update(1)

            # save the processing_preds into predictions_file
            if batch_count % save_interval == 0:
                processing_preds = save_preds_to_file(processing_preds, predictions_file, device)

    save_preds_to_file(processing_preds, predictions_file)
    print(f"Final results saved to {predictions_file}")

def main():
    config = setup_main()
    device = torch.device(config.device)

    val_loaders = create_loader(
        create_dataset(config),
        [None],
        batch_size=[config.batch_size],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[collate_fn],
    )

    for val_loader in val_loaders:
        inference(config.model, val_loader, device, config)

if __name__ == "__main__":
    main()
