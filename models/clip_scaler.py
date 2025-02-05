"""
Adaptation of OpenAI's CLIP.
Requires:
- pip install git+https://github.com/openai/CLIP.git

.. note::
    Checkpoints are loaded from the OpenAI repository.
    * RN50: "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
    * RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"
    * RN50x4: "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt"
    * RN50x16: "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt"
    * RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt"
    * ViT-B/32: "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    * ViT-B/16: "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
    * ViT-L/14: "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
    * ViT-L/14@336px: "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
"""
import random

import torch
import torch.nn as nn


from utils import binary_to_boolean_type
try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")

from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.conf import get_device
from copy import deepcopy
from torch import optim
import torch.func as func
from typing import Iterable, Tuple
from tqdm import tqdm
import gc
import numpy as np
import open_clip
from templates import get_templates
from pathlib import Path
import os


def build_classification_head(model, dataset, offset, eval=False):
    template = get_templates(dataset.NAME)
    classnames = dataset.class_names

    clip_model_open, _ = clip.load(model.args.clip_backbone, device=torch.device(model.args.device))
    #clip_model_open, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=model.args.device)

    clip_model_open.to(dtype=torch.float32)
    clip_model_open.eval()

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []

        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(model.device)  # tokenize
            embeddings = clip_model_open.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(model.device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= 100.
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
    if eval:
        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights[:][:offset[1]])
    else:
        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights[:][offset[0]:offset[1]])

    classification_head.requires_grad_(False)

    return classification_head

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias, device=self.weight.device))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


class FinalModel(nn.Module):
    @torch.no_grad()
    def __init__(self, clip_model, dataset: ContinualDataset, args) -> None:
        super().__init__()
        self.dataset = dataset
        clip_model.to(dtype=torch.float32)

        self.visual_encoder = deepcopy(clip_model.visual)
        self.dtype = torch.float32
        self.args = args

        self.classes = self.dataset.get_class_names()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.task_id = 0

    def forward(self, x, idx_classes=None):
        image_features = self.visual_encoder(x.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


class CLIP(ContinualModel):
    """STATIC Continual Learning with CLIP"""
    NAME = 'clip_scaler'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(lr=0, n_epochs=0)  # disable training by default
        parser.add_argument('--clip_backbone', type=str, default='ViT-L/14',
                            choices=list(clip.available_models()),
                            help='Backbone architecture for CLIP')
        parser.add_argument('--save_predictions', type=binary_to_boolean_type, default=0,
                            help='Whether to save predictions of the TRAINING set after each task')
        parser.add_argument('--use_templates', type=binary_to_boolean_type, default=0,
                            help='Whether to use prompt templates for CLIP. NOTE: Datasets NEED to have a `get_prompt_templates` method implemented.')
        parser.add_argument('--use_heads', type=binary_to_boolean_type, default=0,
                            help='Whether to use prompt templates to build classification heads for CLIP. NOTE: Datasets NEED to have a `get_prompt_templates` method implemented.')
        parser.add_argument('--test_single_task',  type=binary_to_boolean_type, default=0, help='Set to 1 to test single tasks')
        parser.add_argument('--scaler_value', type=float, default=1.0, help='define the coefficient used to scale the task vectors')
        parser.add_argument('--epochs', type=int, help="number of training epochs used during finetuning")
        parser.add_argument('--tangent',  type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune on the tangent hyperplane')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        _, train_preprocess, val_preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=torch.device('cpu'))

        backbone, _ = clip.load(args.clip_backbone, device=torch.device('cpu'))

        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.net = FinalModel(backbone, dataset, args)
        self.param_names = [name for name, _ in self.net.visual_encoder.named_parameters()]
        for name, param in self.net.named_parameters():
            param.requires_grad = False
        torch.backends.cuda.enable_mem_efficient_sdp(False)


        self.clip_transform = train_preprocess
        self.clip_eval_transform = val_preprocess

        self.predictions = []
        self.original_labels = []

    def begin_task(self, dataset):
        print("BEGIN TASK")
        self.cur_offset = self.compute_offsets(self.current_task)

    def end_task(self, dataset: ContinualDataset) -> None:  # TODO  set the model in eval mode
        print(f"Current task: {self.current_task}")
        backbone, _ = clip.load(self.args.clip_backbone, device=torch.device(self.args.device))
        backbone.to(dtype=torch.float32)
        self.cls_head = build_classification_head(self, dataset, self.cur_offset, eval=True)

        tv_path = Path(f"./cache/{self.args.clip_backbone}_{self.args.dataset}_{self.N_TASKS}_{self.args.optimizer}_{self.args.lr}_{self.args.optim_wd}_{self.args.n_epochs}_{self.args.tangent}/{self.current_task}.pt")
        print(tv_path)
        task_vector_dict = torch.load(tv_path)

        if self.args.test_single_task:
            self.merged_params = task_vector_dict
            print("Media parametri aggiornata single task.")
        else:
            if self.current_task > 0:
                for key in self.merged_params:
                    self.merged_params[key].data = self.merged_params[key].data + task_vector_dict[key].data
                print("Media parametri aggiornata.")
            else:
                self.merged_params = task_vector_dict
                print("Media parametri aggiornata.")

        self.net.visual_encoder = None
        self.net.visual_encoder = backbone.visual
        if not self.args.tangent:
            for name, param in self.net.visual_encoder.named_parameters():
                if name in self.merged_params:
                    param.data = param.data + (self.merged_params[name].data / (self.current_task + 1))
        else:
            need_4_name = deepcopy(self.merged_params)
            self.tangent_4_forward = []
            for key in need_4_name:
                need_4_name[key] = need_4_name[key].data / (self.current_task + 1)
                self.tangent_4_forward.append(need_4_name[key])

        torch.cuda.empty_cache()
        self.eval()
        return super().end_task(dataset)

    def forward(self, x):
        if self.args.tangent:
            def func_network(param_values):
                param = {name: param for name, param in zip(self.param_names, param_values)}
                return func.functional_call(self.net.visual_encoder, param, x)

            image_features, jvp = func.jvp(func_network, (tuple(self.net.visual_encoder.parameters()),),
                                           (tuple(self.tangent_4_forward),), )
            image_features = image_features + jvp
        else:
            image_features = func.functional_call(self.net,
                                                  {name: param for name, param in self.net.named_parameters()}, x)
        image_features = nn.functional.normalize(image_features, dim=-1)
        similarity = self.cls_head(image_features)
        return similarity[:, :self.n_seen_classes]