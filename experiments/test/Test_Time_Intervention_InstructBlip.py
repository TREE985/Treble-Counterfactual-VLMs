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
from Code.lavis.models import load_model_and_preprocess
from PIL import Image
from transformers import set_seed
from Treble_Counterfactual_utils.representation_shift import text_shift_IB, visual_shift_IB, cross_modal_shift_IB, Gaussion_Noise_sample_IB, Blank_sample_IB
from Treble_Counterfactual_utils.Test_Time_Intervention import test_time_intervention, remove_vti_layers
from datasets import load_dataset

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_tuple_shape(tup):
    shape = []
    while isinstance(tup, tuple):
        shape.append(len(tup))
        tup = tup[0]
    return shape

def eval_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    tokenizer = InstructBlipProcessor.from_pretrained(args.model_path)
    text_model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path,torch_dtype=torch.float16).to(device)
    vision_model = text_model.vision_model.to(device)

    img_gn, img_bk, input_ids = Gaussion_Noise_sample_IB(args, image_processor, tokenizer)
    _, _, input_ids_bk = Blank_sample_IB(args, image_processor, tokenizer)
    torch.cuda.empty_cache()

    NDE_visual, _ = visual_shift_IB(model, img_gn, rank=args.rankk)
    NV = NDE_visual[1:]
    test_time_intervention(model.model.vision_tower.vision_tower.vision_model, torch.stack([NV], dim=1).cuda(), alpha=[args.NV])

    NDE_text, _ = text_shift_IB(model, input_ids, img_gn, rank=args.rank)
    NT = NDE_text[1:]
    test_time_intervention(model, torch.stack([NT], dim=1).cuda(), alpha=[args.NT])

    NDE_visula_text, _ = cross_modal_shift_IB(model, input_ids_bk, img_bk, rank=args.rank)
    NVT = NDE_visula_text[1:]
    test_time_intervention(model, torch.stack([NVT], dim=1).cuda(), alpha=[args.NVT])
    torch.cuda.empty_cache()
    import os

    test_set = args.test_set
    if test_set == 'mmhal':
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        dataset = load_dataset("MMHal-Bench")['test']
        ans_file = open(answers_file, "w")
        for img_id in tqdm(range(len(dataset)), desc="Processing image"):
            image_path = dataset[img_id]['image_path']
            raw_image = load_image(image_path)
            prompt = dataset[img_id]['question']
            image_tensor = image_processor["eval"](raw_image).unsqueeze(0)
            input_ids = tokenizer(text=prompt, return_tensors="pt")
            img_save = {}
            input_ids2 = tokenizer(images=raw_image, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = text_model.generate(
                        **input_ids2,
                        do_sample=False,
                        num_beams=5,
                        max_length=256,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                        )
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print(generated_text)
            img_save["question_type"] = dataset[img_id]["question_type"]
            img_save["question_topic"] = dataset[img_id]["question_topic"]
            img_save["image_id"] = dataset[img_id]["image_id"]
            img_save["image_src"] = dataset[img_id]["image_src"]
            img_save["image_content"] = dataset[img_id]["image_content"]
            img_save["question"] = dataset[img_id]["question"]
            img_save["gt_answer"] = dataset[img_id]["gt_answer"]
            img_save["model_answer"] = generated_text
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
                prompt = img['text']
                input_ids2 = tokenizer(images=raw_image, text=prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = text_model.generate(
                            **input_ids2,
                            do_sample=False,
                            num_beams=5,
                            max_length=256,
                            min_length=1,
                            top_p=0.9,
                            repetition_penalty=1.5,
                            length_penalty=1.0,
                            temperature=1,
                            )
                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                print(generated_text)
                print('output',generated_text)
                img['answer'] = generated_text
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {save_file}")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--coco_path", type=str, default="",help="Path for coco dataset of POPE")
    parser.add_argument("--save_path", type=str, default="", help="Path for answers to POPE")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--NV", type=float, default=0.9)
    parser.add_argument("--NVT", type=float, default=0.9)
    parser.add_argument("--NT", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_ratio", type=float, default=0.99)
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--test_set", type=str)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)