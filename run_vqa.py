from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch, json
import os
from tqdm import tqdm
import math

MODEL_PATH =  "./models/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
min_pixels = 256*28*28
max_pixels_list = [23*28*23*28, 35*28*35*28, 46*28*46*28, 58*28*58*28]
#[644*644,980*980,1288*1288,1624*1624]
for max_pixels in max_pixels_list:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)
    # TASK = "KBSR"
    OUTPUT_PATH = "./predict_ICCV_VQA_{}.json".format(math.sqrt(max_pixels))
    DATASET_DIR = "./dataset/images"

    with open('./dataset/VQA-SA-question.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)


    for data in tqdm(datas, desc="进度"):
        question = data['question']
        img_url1 = data['image_path']
        img_url = data['image_path'].replace('images\\', '')
        img_path = os.path.join(DATASET_DIR,img_url)
        try:
            if not os.path.exists(img_path):
                continue
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            content = {
            'image_path':img_url1,
            "question": question,
            "result": output_text[0]}
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(content, ensure_ascii=False) + '\n')
        except:
            continue
