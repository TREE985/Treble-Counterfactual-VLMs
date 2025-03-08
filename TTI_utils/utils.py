import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import math
from transformers import set_seed
import random
from .pca import PCA
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import List, Tuple


# Instructblip
def process_image_IB(image_processor, image_raw):
    answer = image_processor["eval"](image_raw).unsqueeze(0)
    return answer

def text_shift_IB(text_model, inputs, image_tensor, rank=1):
    hidden_states = get_hiddenstates_IB(text_model, inputs, image_tensor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data)
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def visual_shift_IB(vision_model, image_tensor, rank=1):

    hidden_states = get_visual_hiddenstates_IB(vision_model, image_tensor)
    n_layers, n_tokens,_ = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all,dim=1)[:]
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction

def cross_modal_shift_IB(text_model, inputs, image_tensor, rank=1):
    hidden_states = get_hiddenstates_bk_IB(text_model, inputs, image_tensor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data)
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def get_prompts_IB(args, tokenizer, data_demos, question):
    from transformers import InstructBlipProcessor
    processor = InstructBlipProcessor.from_pretrained(args.model_path)
    input_ids_positive = []
    input_ids_negative = []
    for k in data_demos:
        image_path = os.path.join(args.data_file, 'train2014', k['image'])
        image_raw = Image.open(image_path).convert("RGB")
        input_ids_positive.append(processor(images=image_raw, text=question + k['value'], return_tensors="pt"))
        input_ids_negative.append(processor(images=image_raw, text=question + k['h_value'], return_tensors="pt"))
    inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
    inputs = tuple(inputs)
    return inputs

def get_prompts_bk_IB(args, tokenizer, data_demos, question):
    input_ids_positive = []
    input_ids_negative = []
    for k in data_demos:
        image_path = os.path.join(args.data_file, 'train2014', k['image'])
        image_raw = Image.open(image_path).convert("RGB")
        input_ids_positive.append(tokenizer(images=image_raw, text=question + k['value'], return_tensors="pt"))
        input_ids_negative.append(tokenizer(images=image_raw, text=question + k['value'], return_tensors="pt"))
    inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
    inputs = tuple(inputs)
    return inputs

def Gaussion_Noise_sample_IB(args, image_processor, tokenizer, patch_size = 14, file_path = 'experiments/data/image_demos.jsonl'):
    # Initialize a list to store the JSON objects
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.sample_num)
    inputs_images_gn = []
    inputs_images_blank = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image_IB(image_processor, image_raw)
        image_tensor_cd_all_trials = []
        image_tensor_blank_all_trials = []
        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = add_gaussian_noise(image_tensor,noise_step = 500)
            image_tensor_blank = add_blank(image_tensor)    
            image_tensor_cd_all_trials.append(image_tensor_cd)
            image_tensor_blank_all_trials.append(image_tensor_blank)
        inputs_images_gn.append([image_tensor_cd_all_trials, image_tensor])
        inputs_images_blank.append([image_tensor_blank_all_trials, image_tensor])
    input_ids = get_prompts_IB(args, tokenizer, data_demos, question)
    return inputs_images_gn, inputs_images_blank, input_ids

def Blank_sample_IB(args, image_processor, tokenizer, patch_size = 14, file_path = 'experiments/data/image_demos.jsonl'):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.sample_num)
    inputs_images_gn = []
    inputs_images_blank = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image_IB(image_processor, image_raw)
        image_tensor_cd_all_trials = []
        image_tensor_blank_all_trials = []
        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = add_gaussian_noise(image_tensor, noise_step=200)
            image_tensor_blank = add_blank(image_tensor)    
            image_tensor_cd_all_trials.append(image_tensor_cd)
            image_tensor_blank_all_trials.append(image_tensor_blank)
        inputs_images_gn.append([image_tensor_cd_all_trials, image_tensor])
        inputs_images_blank.append([image_tensor_blank_all_trials, image_tensor])
    input_ids = get_prompts_bk_IB(args, tokenizer, data_demos, question)
    return inputs_images_gn, inputs_images_blank, input_ids

def get_hiddenstates_IB(text_model, inputs, image_tensor):
        h_all = []
        vision_model = text_model
        with torch.no_grad():
            for example_id in range(len(inputs)):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs[example_id])):
                    if image_tensor is None:
                        output = vision_model.language_model(
                                **inputs[example_id][style_id],
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                        h = output.hidden_states
                    else:
                        output = vision_model(
                                    pixel_values=image_tensor[example_id][-1].half().to(vision_model.device),
                                    qformer_input_ids=inputs[example_id][style_id]['qformer_input_ids'].to(vision_model.device),
                                    input_ids=inputs[example_id][style_id]['input_ids'].to(vision_model.device),
                                    # use_cache=False,
                                    output_hidden_states=True,
                                    return_dict=True)
                        h = output.language_model_outputs.hidden_states

                    embedding_token = []
                    for layer in range(len(h)):
                        embedding_token.append(h[layer][:,-1].detach().cpu())
                    
                    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all

def get_visual_hiddenstates_IB(vision_model, image_tensor):
    h_all = []
    with torch.no_grad():
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []
            for style_id in range(len(image_tensor[example_id])):
                if isinstance(image_tensor[example_id][style_id], list):
                    h = []
                    for image_tensor_ in image_tensor[example_id][style_id]:

                        h_ = vision_model(pixel_values=image_tensor_.cuda(),
                                              output_hidden_states=True)

                        h.append(h_["hidden_states"])

                    h = average_tuples(h)

                else:
                    h_ = vision_model(pixel_values=image_tensor[example_id][style_id].cuda(),
                                         output_hidden_states=True)
                    h = h_["hidden_states"]
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
    del h, embedding_token
    return h_all

def get_hiddenstates_bk_IB(text_model, inputs, image_tensor):
        vision_model = text_model
        h_all = []
        with torch.no_grad():
            for example_id in range(len(inputs)):
                embeddings_for_all_styles= []
                if isinstance(image_tensor[example_id][0], list):
                    h_neg = []
                    for image_tensor_ in image_tensor[example_id][0]:
                        output = vision_model(
                                    pixel_values=image_tensor_.half().to(vision_model.device),
                                    qformer_input_ids=inputs[example_id][-1]['qformer_input_ids'].to(vision_model.device),
                                    input_ids=inputs[example_id][-1]['input_ids'].to(vision_model.device),
                                    output_hidden_states=True,
                                    return_dict=True)
                        h_neg_ = output.language_model_outputs.hidden_states
                        h_neg.append(h_neg_)
                    h_neg = average_tuples(h_neg)
                else:
                    output = vision_model(
                                pixel_values=image_tensor[example_id][0].half().to(vision_model.device),
                                qformer_input_ids=inputs[example_id][0]['qformer_input_ids'].to(vision_model.device),
                                input_ids=inputs[example_id][-1]['input_ids'].to(vision_model.device),
                                output_hidden_states=True,
                                return_dict=True)
                    h_neg_ = output.language_model_outputs.hidden_states
                    h_neg = h_neg_
                embedding_token = []
                for layer in range(len(h_neg)):
                    embedding_token.append(h_neg[layer][:,-1].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)

                h_pos_ = vision_model.language_model(
                            input_ids=inputs[example_id][-1]['input_ids'].to(vision_model.device),
                            output_hidden_states=True,
                            return_dict=True)

                h_pos = h_pos_.hidden_states
                embedding_token = []
                for layer in range(len(h_pos)):
                    embedding_token.append(h_pos[layer][:,-1].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all



# llava
def process_image(image_processor, image_raw):

    answer = image_processor(image_raw)

    if 'pixel_values' in answer:
        answer = answer['pixel_values'][0]

    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)

    elif isinstance(answer, torch.Tensor):
        return answer
    else:
        raise ValueError("Unexpected output format from image_processor.")
    return answer

def visual_shift(model, image_tensor, rank=1, model_is_llaval=True):
    hidden_states = get_visual_hiddenstates(model, image_tensor, model_is_llaval = model_is_llaval)
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)

    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all,dim=1)[:] # n_token (no CLS token) x n_demos x D
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction

def text_shift(model, inputs, image_tensor, rank=1):
    hidden_states = get_hiddenstates(model, inputs, image_tensor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    for demonstration_id in range(num_demonstration):

        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float()) 
    eval_data =  pca.transform(fit_data.float()) 
    h_pca = pca.inverse_transform(eval_data) 
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def cross_modal_shift(model, inputs, image_tensor, rank=1):
    hidden_states = get_hiddenstates_bk(model, inputs, image_tensor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data)
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def get_prompts(args, model, tokenizer, data_demos, question, model_is_llaval=True):
    if model_is_llaval:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        qs_pos = question
        qs_neg = question

        if hasattr(model.config, 'mm_use_im_start_end'):

            if model.config.mm_use_im_start_end:
                qs_pos = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_pos
            else:
                qs_pos = DEFAULT_IMAGE_TOKEN + '\n' + qs_pos

            if model.config.mm_use_im_start_end:
                qs_neg = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_neg
            else:
                qs_neg = DEFAULT_IMAGE_TOKEN + '\n' + qs_neg

            conv_pos = conv_templates[args.conv_mode].copy()
            conv_pos.append_message(conv_pos.roles[0], qs_pos)
            conv_pos.append_message(conv_pos.roles[1], None)
            conv_neg = conv_templates[args.conv_mode].copy()
            conv_neg.append_message(conv_neg.roles[0], qs_neg)
            conv_neg.append_message(conv_neg.roles[1], None)


            prompts_positive  = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            prompts_negative  = [conv_neg.get_prompt() + k['h_value'] for k in data_demos]

            input_ids_positive = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_positive]
            input_ids_negative = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_negative]

        else:
            from transformers import InstructBlipProcessor
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

            input_ids_positive = []
            input_ids_negative = []

            for k in data_demos:
                image_path = os.path.join(args.data_file, 'train2014', k['image'])

                image_raw = Image.open(image_path).convert("RGB")
                input_ids_positive.append(processor(images=image_raw, text=question + k['value'], return_tensors="pt").to(model.device))
                input_ids_negative.append(processor(images=image_raw, text=question + k['h_value'], return_tensors="pt").to(model.device))

        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    else:

        prompts_positive = []
        prompts_negative = []

        for k in data_demos:
            image_path = os.path.join(args.data_file, 'train2014', k['image'])    
            prompts_positive.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['value']}]))
            prompts_negative.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['h_value']}]))

        input_ids_positive = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_positive]
        input_ids_negative = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_negative]
        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    return inputs

def get_hiddenstates(model, inputs, image_tensor):
        h_all = []
        with torch.no_grad():
            for example_id in range(len(inputs)):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs[example_id])):
                    if image_tensor is None:
                        h = model(
                                **inputs[example_id][style_id],
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                    else:
                        h = model(
                                inputs[example_id][style_id],
                                images=image_tensor[example_id][-1].unsqueeze(0).half(),
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True).hidden_states

                    embedding_token = []
                    for layer in range(len(h)):
                        embedding_token.append(h[layer][:,-1].detach().cpu())
                    
                    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all

def average_tuples(tuples: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    # Check that the input list is not empty
    if not tuples:
        raise ValueError("The input list of tuples is empty.")

    # Check that all tuples have the same length
    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")

    # Initialize a list to store the averaged tensors
    averaged_tensors = []

    # Iterate over the indices of the tuples
    for i in range(n):
        # Stack the tensors at the current index and compute the average
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)

    # Convert the list of averaged tensors to a tuple
    averaged_tuple = tuple(averaged_tensors)

    return averaged_tuple

def get_visual_hiddenstates(model, image_tensor, model_is_llaval=True):
    h_all = []
    with torch.no_grad():
        if model_is_llaval:
            try:
                vision_model = model.model.vision_tower.vision_tower.vision_model
            except:
                vision_model = model.vision_model
        else:
            vision_model = model.transformer.visual
            model.transformer.visual.output_hidden_states = True
            
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []
            for style_id in range(len(image_tensor[example_id])):
                if isinstance(image_tensor[example_id][style_id], list):
                    h = []
                    for image_tensor_ in image_tensor[example_id][style_id]:
                        if model_is_llaval:
                            h_ = vision_model(
                                image_tensor_.unsqueeze(0).half().cuda(),
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                        else:
                            _, h_ = vision_model(
                                image_tensor_.unsqueeze(0).cuda())
                        h.append(h_)
                    h = average_tuples(h)
                else:
                    if model_is_llaval:
                        h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).half().cuda(),
                            output_hidden_states=True,
                            return_dict=True).hidden_states

                    else:
                        _, h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda())
                
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
        if not model_is_llaval:
            model.transformer.visual.output_hidden_states = False

    del h, embedding_token

    return h_all

def add_gaussian_noise(image_tensor, noise_step = 500):
    num_steps = 1000  # Number of diffusion steps
    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 
    
    return image_tensor_cd

def add_blank(image_tensor):
    return torch.zeros_like(image_tensor)

def Gaussion_Noise_sample(args, image_processor, model, tokenizer, patch_size = 14, file_path = 'experiments/data/image_demos.jsonl', model_is_llaval=True):
    # Initialize a list to store the JSON objects
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)

    inputs_images_gn = []
    inputs_images_blank = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(image_processor, image_raw)
        image_tensor_cd_all_trials = []
        image_tensor_blank_all_trials = []

        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = add_gaussian_noise(image_tensor,noise_step = 500)
            image_tensor_blank = add_blank(image_tensor)    
            image_tensor_cd_all_trials.append(image_tensor_cd)
            image_tensor_blank_all_trials.append(image_tensor_blank)

        inputs_images_gn.append([image_tensor_cd_all_trials, image_tensor])
        inputs_images_blank.append([image_tensor_blank_all_trials, image_tensor])

    input_ids = get_prompts(args, model, tokenizer, data_demos, question, model_is_llaval=model_is_llaval)
    
    return inputs_images_gn, inputs_images_blank, input_ids

def get_prompts_bk(args, model, tokenizer, data_demos, question, model_is_llaval=True):
    if model_is_llaval:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        qs_pos = question
        qs_neg = question

        if hasattr(model.config, 'mm_use_im_start_end'):

            if model.config.mm_use_im_start_end:
                qs_pos = DEFAULT_IM_START_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_pos
            else:
                qs_pos = qs_pos

            if model.config.mm_use_im_start_end:
                qs_neg = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_neg
            else:
                qs_neg = DEFAULT_IMAGE_TOKEN + '\n' + qs_neg

            conv_pos = conv_templates[args.conv_mode].copy()
            conv_pos.append_message(conv_pos.roles[0], qs_pos)
            conv_pos.append_message(conv_pos.roles[1], None)
            conv_neg = conv_templates[args.conv_mode].copy()
            conv_neg.append_message(conv_neg.roles[0], qs_neg)
            conv_neg.append_message(conv_neg.roles[1], None)


            prompts_positive  = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            prompts_negative  = [conv_neg.get_prompt() + k['value'] for k in data_demos]

            input_ids_positive = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_positive]
            input_ids_negative = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_negative]

        else:
            from transformers import InstructBlipProcessor
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

            input_ids_positive = []
            input_ids_negative = []

            for k in data_demos:
                image_path = os.path.join(args.data_file, 'train2014', k['image'])

                image_raw = Image.open(image_path).convert("RGB")
                input_ids_positive.append(processor(images=image_raw, text=question + k['value'], return_tensors="pt").to(model.device))
                input_ids_negative.append(processor(images=image_raw, text=question + k['value'], return_tensors="pt").to(model.device))

        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    else:

        prompts_positive = []
        prompts_negative = []

        for k in data_demos:
            image_path = os.path.join(args.data_file, 'train2014', k['image'])    
            prompts_positive.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['value']}]))
            prompts_negative.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['value']}]))

        input_ids_positive = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_positive]
        input_ids_negative = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_negative]
        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    return inputs

def Blank_sample(args, image_processor, model, tokenizer, patch_size = 14, file_path = 'experiments/data/image_demos.jsonl', model_is_llaval=True):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)
    inputs_images_gn = []
    inputs_images_blank = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(image_processor, image_raw)
        image_tensor_cd_all_trials = []
        image_tensor_blank_all_trials = []
        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = add_gaussian_noise(image_tensor, mask_index, noise_step=500)
            image_tensor_blank = add_gaussian_noise(image_tensor, noise_step=200)    
            image_tensor_cd_all_trials.append(image_tensor_cd)
            image_tensor_blank_all_trials.append(image_tensor_blank)
        inputs_images_gn.append([image_tensor_cd_all_trials, image_tensor])
        inputs_images_blank.append([image_tensor_blank_all_trials, image_tensor])
    input_ids = get_prompts_bk(args, model, tokenizer, data_demos, question, model_is_llaval=model_is_llaval)
    return inputs_images_gn, inputs_images_blank, input_ids

def get_hiddenstates_bk(model, inputs, image_tensor):
        h_all = []
        with torch.no_grad():
            for example_id in range(len(inputs)):
                embeddings_for_all_styles= []
                if isinstance(image_tensor[example_id][0], list):
                    h_neg = []
                    for image_tensor_ in image_tensor[example_id][0]:
                        h_neg_ = model(
                                    inputs[example_id][-1],
                                    images=image_tensor_.unsqueeze(0).half(),
                                    use_cache=False,
                                    output_hidden_states=True,
                                    return_dict=True).hidden_states
                        h_neg.append(h_neg_)
                    h_neg = average_tuples(h_neg)
                else:
                    h_neg = model(
                                inputs[example_id][0],
                                images=image_tensor[example_id][0].unsqueeze(0).half(),
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                embedding_token = []
                for layer in range(len(h_neg)):
                    embedding_token.append(h_neg[layer][:,-1].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)
                h_pos = model(
                            inputs[example_id][-1],
                            
                            use_cache=False,
                            output_hidden_states=True,
                            return_dict=True).hidden_states
                embedding_token = []
                for layer in range(len(h_pos)):
                    embedding_token.append(h_pos[layer][:,-1].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all

