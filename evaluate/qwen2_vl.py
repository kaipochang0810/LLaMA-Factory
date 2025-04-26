from qwen_vl_utils import process_vision_info
from PIL import Image

def preprocess_image_nearest(image_path: str, image_resolution: int, perform_resize: bool = False) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            original_width, original_height = image.width, image.height
            max_dimension = max(original_width, original_height)

            if max_dimension > image_resolution:
                resize_factor = image_resolution / max_dimension
                new_width = int(original_width * resize_factor)
                new_height = int(original_height * resize_factor)
            else:
                new_width, new_height = original_width, original_height

            if perform_resize:
                resized_image = image.resize((new_width, new_height), resample=Image.NEAREST)
                if image.mode != "RGB":
                    resized_image = resized_image.convert("RGB")
                return new_width, new_height, resized_image

    except Exception as e:
        raise IOError(f"Can not open the image {image_path}: {e}")

    return new_height, new_width



def build_message(prompt, video_fp):
    message = []
    user_prompt = [
        {"type": "video", "video": video_fp, "nframes": 64, "resized_height": 384, "resized_width": 384+28},  # TODO: Hardcode for saving object patch preprocessing time
        {"type": "text", "text": prompt}
    ]
    
    message.append({"role": "user",  "content": user_prompt})
    return message

def get_response_qwen(model, processor, prompts, video_fps, config, eval_name):
            
    messages = []
    for prompt, video_fp in zip(prompts, video_fps):
        messages.append(build_message(prompt, video_fp))

    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs
    )
    inputs = inputs.to("cuda")
    # inputs = dict_keys(['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 'second_per_grid_ts'])
    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts