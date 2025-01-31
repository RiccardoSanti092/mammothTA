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

from adamow import AdamW

def get_params(net, features=True, classifier=False, offset_1=-1, offset_2=-1) -> torch.Tensor:
    params = []
    for name, param in net.named_parameters():
        if "head" in name:
            if classifier:
                assert (offset_1 > -1 and offset_2 > -1)
                params.append(param[offset_1:offset_2].view(-1))
        elif features:
            params.append(param.view(-1))

    if len(params):
        return torch.cat(params)
    else:
        return torch.tensor([0.])


def set_params(net, new_params: torch.Tensor, features=True, classifier=False, offset_1=-1, offset_2=-1) -> None:
    progress = 0
    for name, param in net.named_parameters():
        if "head" in name:
            if classifier:
                assert (offset_1 > -1 and offset_2 > -1)
                cur_size = torch.tensor(param.data[offset_1:offset_2].size()).prod()
                param.data[offset_1:offset_2] = new_params[progress: progress + cur_size].view(
                    param.data[offset_1:offset_2].size())
                progress += cur_size
        elif features:
            cur_size = torch.tensor(param.size()).prod()
            cand_params = new_params[progress: progress + cur_size].view(param.size())
            param.data = cand_params
            progress += cur_size


def get_delta_w_backbone(named_params, delta_w, delta_w_names, peft_type, device, vera_B=None, vera_A=None, vera_r=None):
    params = []
    for name, param in named_params():
        if "head" not in name:
            if name in delta_w_names:
                index = delta_w_names.index(name)
                if peft_type == "lora":
                    cur_delta_w = delta_w[index][0] @ delta_w[index][1]
                elif peft_type == "ia3":
                    cur_delta_w = delta_w[index] * param.data
                elif peft_type == "full":
                    cur_delta_w = delta_w[index]
                elif peft_type == "vera":
                    cur_delta_w = (delta_w[index][0] * vera_B[:param.shape[0], :vera_r]) @ (delta_w[index][1] * vera_A[:vera_r, :param.shape[1]])
                params.append(cur_delta_w.view(-1).to(device))
            else:
                params.append(torch.zeros_like(param).view(-1).to(device))

    if len(params):
        return torch.cat(params)
    else:
        return torch.tensor([0.]).to(device)


def get_delta_w_parameterlist(named_params, delta_w, delta_w_names, peft_type, device, vera_B=None, vera_A=None, vera_r=None):
    params = []
    for name, param in named_params():
        if name in delta_w_names:
            index = delta_w_names.index(name)
            if peft_type == "lora":
                cur_delta_w = delta_w[index][0] @ delta_w[index][1]
            elif peft_type == "ia3":
                cur_delta_w = delta_w[index] * param.data
            elif peft_type == "full":
                cur_delta_w = delta_w[index]
            elif peft_type == "vera":
                cur_delta_w = (delta_w[index][0] * vera_B[:param.shape[0], :vera_r]) @ (delta_w[index][1] * vera_A[:vera_r, :param.shape[1]])
            params.append(cur_delta_w.to(device))
        else:
            params.append(torch.zeros_like(param).to(device))

    return params

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

    @torch.no_grad()
    def compute_text_embeddings(self, clip_model, classes):

        if self.args.use_templates:
            templates = self.dataset.get_prompt_templates()
            text_inputs = []
            for t in templates:
                t_inputs = torch.cat([clip.tokenize(t.format(c)) for c in classes]).to(torch.device("cpu"))
                t_inputs = clip_model.encode_text(t_inputs)
                t_inputs /= t_inputs.norm(dim=-1,
                                          keepdim=True)  # double normalization if use templates is expected (see https://github.dev/KaiyangZhou/CoOp)

                text_inputs.append(t_inputs)
            text_features = torch.stack(text_inputs).mean(0)
        else:
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(torch.device("cpu"))
            text_features = clip_model.encode_text(text_inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu()

    def forward(self, x, idx_classes=None):
        image_features = self.visual_encoder(x.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

class CLIP(ContinualModel):
    """STATIC Continual Learning with CLIP"""
    NAME = 'clip_ft'
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
        parser.add_argument('--ft_linears', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune linear layers')
        parser.add_argument('--ft_attention', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune attention layers')
        parser.add_argument('--ft_ln', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune layer norm')
        parser.add_argument('--ft_class_embed', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune class embedding layers')
        parser.add_argument('--ft_pos_embed', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune posistional embedding')
        parser.add_argument('--ft_proj', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune projection layers')
        parser.add_argument('--ft_conv', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune convolutional layers')
        parser.add_argument('--tangent',  type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune on the tangent hyperplane')
        parser.add_argument('--chunks', type=int, default=1, help='chose how many chunks for vitual batch size')

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

        print("Updating the following layers:")
        if self.args.ft_linears:
            print("- linears")
        if self.args.ft_attention:
            print("- attention")
        if self.args.ft_class_embed:
            print("- class embeddings")
        if self.args.ft_conv:
            print("- convolutional")
        if self.args.ft_ln:
            print("- layer norm")
        if self.args.ft_pos_embed:
            print("- positional embedding")
        if self.args.ft_proj:
            print("- projection")

    def begin_epoch(self, epoch: int, dataset: ContinualDataset) -> None:
        torch.cuda.empty_cache()

    #def end_epoch(self, epoch: int, dataset: ContinualDataset) -> None:
        #self.scheduler.step()

    def begin_task(self, dataset):
        torch.cuda.empty_cache()
        dataset.test_loaders[-1].dataset.transform = self.clip_eval_transform
        dataset.train_loader.dataset.transform = self.clip_transform
        self.cur_offset = self.compute_offsets(self.current_task)

        if isinstance(dataset.N_CLASSES_PER_TASK, int):
            self.cpt = dataset.N_CLASSES_PER_TASK
        else:
            self.cpt = dataset.N_CLASSES_PER_TASK[-1]

        if self.current_task != 0:
            self.net.task_id += 1

        self.cls_head = build_classification_head(self, dataset, self.cur_offset)

        print("\nRELOADING CLIP VISUAL ENCODER")
        self.net.visual_encoder = None
        #backbone, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=torch.device(self.args.device))

        backbone, _ = clip.load(self.args.clip_backbone, device=torch.device(self.args.device))

        self.net.visual_encoder = backbone.visual
        self.net.visual_encoder.to(dtype=torch.float32)
        for param in self.net.visual_encoder.parameters():
            param.requires_grad = False
        print("\nCLIP VISUAL ENCODER RELOADED\n\n")

        self.delta_w = []
        self.delta_w_names = []
        for name, param in self.net.visual_encoder.named_parameters():
            if self.args.ft_linears and "mlp" in name:
                self.delta_w.append(torch.zeros_like(param, dtype = torch.float32, requires_grad = True, device = self.args.device))
                self.delta_w_names.append(name)
            elif self.args.ft_attention and "attn" in name:
                self.delta_w.append(torch.zeros_like(param, dtype = torch.float32, requires_grad = True, device = self.args.device))
                self.delta_w_names.append(name)
            elif self.args.ft_class_embed and "class_embed" in name:
                self.delta_w.append(torch.zeros_like(param, dtype = torch.float32, requires_grad = True, device = self.args.device))
                self.delta_w_names.append(name)
            elif self.args.ft_conv and "conv" in name:
                self.delta_w.append(torch.zeros_like(param, dtype = torch.float32, requires_grad = True, device = self.args.device))
                self.delta_w_names.append(name)
            elif self.args.ft_ln and "ln" in name:
                self.delta_w.append(torch.zeros_like(param, dtype = torch.float32, requires_grad = True, device = self.args.device))
                self.delta_w_names.append(name)
            elif self.args.ft_pos_embed and "positional_embedding" in name:
                self.delta_w.append(torch.zeros_like(param, dtype = torch.float32, requires_grad = True, device = self.args.device))
                self.delta_w_names.append(name)
            elif self.args.ft_proj and "proj" in name:
                self.delta_w.append(torch.zeros_like(param, dtype = torch.float32, requires_grad = True, device = self.args.device))
                self.delta_w_names.append(name)

        if self.args.optimizer == 'adamw':
            self.opt = optim.AdamW(self.delta_w, lr=self.args.lr,
                                  weight_decay=self.args.optim_wd)

        else:
            self.opt = optim.SGD(self.delta_w, lr=self.args.lr,
                                 momentum=self.args.optim_mom)

        num_batches = len(dataset.train_loader)
        self.scheduler1 = cosine_lr(self.opt, self.args.lr, 500, self.args.n_epochs * num_batches)

        self.train()
        self.virtual_batch_counter = 0

    def end_task(self, dataset: ContinualDataset) -> None: #TODO  set the model in eval mode
        print(f"Current task: {self.current_task}")

        #backbone, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=torch.device(self.args.device))
        backbone, _ = clip.load(self.args.clip_backbone, device=torch.device(self.args.device))

        backbone.to(dtype=torch.float32)

        self.cls_head = build_classification_head(self, dataset, self.cur_offset, eval=True)

        task_vector_dict = {}
        for i in range(len(self.delta_w)):
            task_vector_dict[self.delta_w_names[i]] = self.delta_w[i]

        if self.args.tangent:
            if self.current_task > 0:
                for key in self.merged_params:
                    self.merged_params[key].data = self.merged_params[key].data + task_vector_dict[key].data
                print("Somma task vector aggiornata.")
            else:
                self.merged_params = task_vector_dict
                print("Somma task vector aggiornata.")
        else:
            if self.current_task > 0:
                for key in self.merged_params:
                    self.merged_params[key].data = self.merged_params[key].data + task_vector_dict[key].data
                print("Media parametri aggiornata.")
            else:
                self.merged_params = task_vector_dict
                print("Media parametri aggiornata.")


        self.opt.zero_grad()
        self.opt = None
        del self.opt, self.delta_w, self.delta_w_names
        gc.collect()

        self.net.visual_encoder = None
        self.net.visual_encoder = backbone.visual
        for name, param in self.net.visual_encoder.named_parameters():
            if name in self.merged_params:
                param.data = param.data + (self.merged_params[name].data / (self.current_task + 1))

        self.tangent_4_forward = []
        for key in self.merged_params:
            self.tangent_4_forward.append(self.merged_params[key])


        torch.cuda.empty_cache()
        self.eval()
        return super().end_task(dataset)



    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        if self.args.tangent:

            def func_network(param_values):
                param = {name: param for name, param in zip(self.delta_w_names, param_values)}
                return func.functional_call(self.net.visual_encoder, param, inputs)

            tunable_params = [p for n, p in self.net.visual_encoder.named_parameters() if n in self.delta_w_names]
            dict_param = {name: param + net_param for name, param, net_param in zip(self.delta_w_names, self.delta_w, tunable_params)}
            params = [param + net_param for name, param, net_param in zip(self.delta_w_names, self.delta_w, tunable_params)]

            image_features, jvp = func.jvp(func_network, (tuple(params),), (tuple(self.delta_w),),)
            image_features = image_features + jvp
        else:

            tunable_params = [p for n, p in self.net.visual_encoder.named_parameters() if n in self.delta_w_names]
            dict_param = {name: param + net_param for name, param, net_param in zip(self.delta_w_names, self.delta_w, tunable_params)}

            image_features = func.functional_call(self.net.visual_encoder, dict_param, inputs)

        image_features = nn.functional.normalize(image_features, dim=-1)
        similarity = self.cls_head(image_features)
        loss = self.loss(similarity, (labels % int(self.N_CLASSES / self.N_TASKS))) / self.args.chunks
        loss.backward()
        self.virtual_batch_counter += 1

        if self.virtual_batch_counter % self.args.chunks == 0:
            torch.nn.utils.clip_grad_norm_(self.delta_w, 1.0)
            self.scheduler1(self.virtual_batch_counter)
            self.opt.step()
            self.opt.zero_grad()

        return loss.item()


    @torch.no_grad()
    def forward(self, x):

        if self.args.tangent:
            def func_network(param_values):
                param = {name: param for name, param in zip(self.param_names, param_values)}
                return func.functional_call(self.net.visual_encoder, param, x)

            image_features, jvp = func.jvp(func_network, (tuple(self.net.visual_encoder.parameters()),), (tuple(self.tangent_4_forward),), )
            image_features = image_features + jvp
        else:
            image_features = func.functional_call(self.net, {name: param for name, param in self.net.named_parameters()}, x)
        image_features = nn.functional.normalize(image_features, dim=-1)
        similarity = self.cls_head(image_features)
        return similarity[:, :self.n_seen_classes]

    def get_debug_iters(self):
        return 128
