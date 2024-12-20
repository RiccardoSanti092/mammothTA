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
from torch import func


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


class FinalModel(nn.Module):
    @torch.no_grad()
    def __init__(self, clip_model, dataset: ContinualDataset, args) -> None:
        super().__init__()
        self.dataset = dataset
        self.visual_encoder = deepcopy(clip_model.visual)
        self.dtype = clip_model.dtype
        self.args = args


        self.classes = self.dataset.get_class_names()
        self.register_buffer("text_features", self.compute_text_embeddings(clip_model, self.classes))
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

    def forward_old(self, x, idx_classes=None):
        image_features = self.visual_encoder(x.type(self.dtype))
        text_features = self.text_features

        if idx_classes is not None:
            text_features = text_features[idx_classes]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * (image_features @ text_features.T)).softmax(dim=-1)

        return similarity


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
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        backbone, clip_transform = clip.load(args.clip_backbone, device=torch.device("cpu"))
        clip.model.convert_weights(backbone)

        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.net = FinalModel(backbone, dataset, args)

        self.param_names = [name for name, _ in self.net.named_parameters()]

        for name, param in self.net.named_parameters():
            param.requires_grad = False

        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.clip_transform = clip_transform

        self.predictions = []
        self.original_labels = []
        self.task_vector_list = []

    def begin_task(self, dataset):

        dataset.test_loaders[-1].dataset.transform = self.clip_transform
        dataset.train_loader.dataset.transform = self.clip_transform

        if self.current_task != 0:
            self.net.task_id += 1

        # PREPARAZIONE PARAMETRI PER TMC   passa dati all'optim
        self.delta_w = []
        self.task_vector = deepcopy(self.net)
        for name, param in self.task_vector.named_parameters():
            param.requires_grad = True
            self.delta_w.append(param)
            self.params_optimizer = self.delta_w



        self.opt = optim.SGD(self.params_optimizer, lr=self.args.lr,
                             momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        del self.delta_w, self.task_vector

        self.train()

    def end_task(self, dataset: ContinualDataset) -> None:
        if self.args.save_predictions:
            self.predictions = torch.cat(self.predictions, dim=0).cpu()
            self.original_labels = torch.cat(self.original_labels, dim=0).cpu()
            torch.save((self.predictions, self.original_labels), f'predictions_{self.args.dataset}_{self.current_task}.pt')
            print(f"Predictions saved for task {self.current_task} in 'predictions_{self.args.dataset}_{self.current_task}.pt'")
            self.predictions = []
            self.original_labels = []

        task_vector_dict = {name: param_finetuned - param_pretrained
                            for ((name, param_pretrained), (param_finetuned))
                            in zip(self.net.named_parameters(), self.params_optimizer)}

        self.task_vector_list.append(task_vector_dict)
        self.merged_parames = {}
        for task_vector in self.task_vector_list:
            for key, tensor in task_vector.items():
                if key in self.merged_parames:
                    self.merged_parames[key] += tensor
                else:
                    self.merged_parames[key] = tensor.clone()

        if self.current_task > 0:
            print("Averaging task vectors")
            for key, tensor in self.merged_parames.items():
                tensor /= (self.current_task + 1)

        for name, param in self.net.named_parameters():
            if name in self.merged_parames:
                self.merged_parames[name] += param.data
            else:
                self.merged_parames[name] = param.data.clone()

        return super().end_task(dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch = None):

        self.opt.zero_grad()
        param = {name: param for name, param in zip(self.param_names, self.params_optimizer)}
        image_features = func.functional_call(self.net, param, inputs)
        text_features = self.net.text_features[torch.unique(labels).tolist()]
        similarity = (100.0 * (image_features @ text_features.T)).softmax(dim=-1)
        loss = self.loss(similarity, (labels % 2))
        loss.backward()
        self.opt.step()

        return loss.item()

    def observe_old(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):

        outputs = self.net(inputs, torch.unique(labels).tolist())
        loss = self.loss(outputs, (labels % 2))
        loss.backward()
        self.opt.step()


    @torch.no_grad()
    def forward(self, x):
        #param = {name: param for name, param in zip(self.param_names, self.params_optimizer)}
        image_features = func.functional_call(self.net, self.merged_parames, x)
        similarity = (100.0 * (image_features @ self.net.text_features.T)).softmax(dim=-1)
        return similarity[:, :self.n_seen_classes]

    @torch.no_grad()
    def forward_old(self, x):
        return self.net(x)[:, :self.n_seen_classes]

