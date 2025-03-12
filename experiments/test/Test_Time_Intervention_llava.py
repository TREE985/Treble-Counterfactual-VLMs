import argparse
import requests
import torch
import os
import json
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import BytesIO, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from transformers import set_seed
from Treble_Counterfactual_utils.representation_shift import text_shift, visual_shift, cross_modal_shift, Gaussion_Noise_sample, Blank_sample
from Treble_Counterfactual_utils.Test_Time_Intervention import test_time_intervention
from datasets import load_dataset

def get_tuple_shape(tup):
    shape = []
    while isinstance(tup, tuple):
        shape.append(len(tup))
        tup = tup[0]
    return shape

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(args):
    import os
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    img_gn, img_bk, input_ids = Gaussion_Noise_sample(args, image_processor, model, tokenizer)
    _, _, input_ids_bk = Blank_sample(args, image_processor, model, tokenizer)

    torch.cuda.empty_cache()

    qs = ''
    if model.config.mm_use_im_start_end: 
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + model.get_vision_tower().num_patches

    img_gn = [img.half().cuda() if isinstance(img, torch.Tensor) else img for img in img_gn]
    img_bk = [img.half().cuda() if isinstance(img, torch.Tensor) else img for img in img_bk]


    NDE_visual, _ = visual_shift(model, img_gn, rank=args.rankk)
    NV = NDE_visual[1:]
    test_time_intervention(model.model.vision_tower.vision_tower.vision_model, torch.stack([NV], dim=1).cuda(), alpha=[args.NV])

    NDE_text, _ = text_shift(model, input_ids, img_gn, rank=args.rank)
    NT = NDE_text[1:]
    test_time_intervention(model, torch.stack([NT],dim=1).cuda(), alpha = [args.NT])

    NDE_visula_text, _ = cross_modal_shift(model, input_ids_bk, img_bk, rank=args.rank)
    NVT = NDE_visula_text[1:]
    test_time_intervention(model, torch.stack([NVT], dim=1).cuda(), alpha=[args.NVT])

    torch.cuda.empty_cache()


    test_set = args.test_set
    if test_set == 'mmhal':
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        dataset = load_dataset("MMHal-Bench")['test']
        ans_file = open(answers_file, "w")
        for img_id in tqdm(range(len(dataset)), desc="Processing image"):
            image_path = dataset[img_id]['image_path']
            raw_image = load_image(image_path)
            qs = dataset[img_id]['question']
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            img_save = {}
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    num_beams=5,
                    max_new_tokens=500,
                    do_sample=args.sample,
                    use_cache=True)
            output_ids = output_ids[output_ids >= 0]
            input_token_len = input_ids.shape[1]-1
            outputs = tokenizer.batch_decode(output_ids[input_token_len:], skip_special_tokens=True)
            sentence = ' '.join(outputs).replace(' .', '.').replace(' ,', ',')
            print('output',sentence)
            img_save["question_type"] = dataset[img_id]["question_type"]
            img_save["question_topic"] = dataset[img_id]["question_topic"]
            img_save["image_id"] = dataset[img_id]["image_id"]
            img_save["image_src"] = dataset[img_id]["image_src"]
            img_save["image_content"] = dataset[img_id]["image_content"]
            img_save["question"] = dataset[img_id]["question"]
            img_save["gt_answer"] = dataset[img_id]["gt_answer"]
            img_save["model_answer"] = sentence
            ans_file.write(json.dumps(img_save) + "\n")
            ans_file.flush()
        ans_file.close()


    elif test_set == 'coco':
        dataset_coco = ['coco_pope_adversarial.json',
                        'coco_pope_random.json',
                        'coco_pope_popular.json']
        full_path = args.coco_path
        save_path = args.save_path
        import os
        os.makedirs(save_path, exist_ok=True)
        answer_suffix = args.answers_file
        for data_name in dataset_coco:
            data_path = full_path + data_name
            save_file = os.path.join(save_path, data_name.replace('.json', f'_{answer_suffix}.json'))
            with open(data_path, 'r', encoding='utf-8') as f:
                dataset = [json.loads(q) for q in f]
            print("Test on ", data_name)
            print("Save in ", save_file)
            for img in tqdm(dataset, desc="Processing image"):
                image_path = img['image']
                raw_image = load_image(image_path)
                qs = img['text']
                cur_prompt = qs
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        num_beams=5,
                        max_new_tokens=256,
                        do_sample=args.sample,
                        use_cache=True)
                output_ids = output_ids[output_ids >= 0]
                input_token_len = input_ids.shape[1]-1
                outputs = tokenizer.batch_decode(output_ids[input_token_len:], skip_special_tokens=True)
                sentence = ' '.join(outputs).replace(' .', '.').replace(' ,', ',')
                print('output',sentence)
                img['answer'] = sentence
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {save_file}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--coco_path", type=str, default="",help="Path for coco dataset of POPE")
    parser.add_argument("--save_path", type=str, default="", help="Path for answers to POPE")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--NV", type=float, default=0.9)
    parser.add_argument("--NVT", type=float, default=0.9)
    parser.add_argument("--NT", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_ratio", type=float, default=0.99)
    parser.add_argument("--num_trials", type=int, default=50)

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)