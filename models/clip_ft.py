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



def get_params(net, features=True, classifier=False, offset_1=-1, offset_2=-1) -> torch.Tensor:
    params = []
    for name, param in net.named_parameters():
        if "head" in name and classifier:
            assert (offset_1 > -1 and offset_2 > -1)
            params.append(param[offset_1:offset_2].view(-1))
        elif features:
            params.append(param.view(-1))

    if len(params):
        return torch.cat(params)
    else:
        return torch.tensor([0.])


def set_params(net, new_params: torch.Tensor, features=True, classifier=False, offset_1=-1, offset_2=-1, ignore_classifier=False) -> None:
    progress = 0
    for name, param in net.named_parameters():
        if "head" in name and classifier:
            if not ignore_classifier:
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


class FinalModel(nn.Module):
    @torch.no_grad()
    def __init__(self, clip_model, dataset: ContinualDataset, args) -> None:
        super().__init__()
        self.dataset = dataset
        self.visual_encoder_0 = deepcopy(clip_model.visual)
        self.visual_encoder_final = deepcopy(clip_model.visual)
        set_params(self.visual_encoder_final, torch.zeros_like(get_params(self.visual_encoder_final)))
        self.dtype = clip_model.dtype
        self.args = args

        self.register_buffer('delta_w_params_history',
                             torch.zeros((dataset.N_TASKS, *get_params(self.final_net).shape)))

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

    def forward(self, x, idx_classes=None):  #TODO: modificare visual_encoder
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

        self.clip_transform = clip_transform

        self.predictions = []
        self.original_labels = []
        print(self.NAME)

    def begin_task(self, dataset):

        dataset.test_loaders[-1].dataset.transform = self.clip_transform
        dataset.train_loader.dataset.transform = self.clip_transform

        if self.current_task != 0:
            self.net.task_id += 1

        self.train()
        self.opt = optim.SGD(self.net.parameters(), lr=self.args.lr,
                             momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)

    def end_task(self, dataset: ContinualDataset) -> None:
        if self.args.save_predictions:
            self.predictions = torch.cat(self.predictions, dim=0).cpu()
            self.original_labels = torch.cat(self.original_labels, dim=0).cpu()
            torch.save((self.predictions, self.original_labels), f'predictions_{self.args.dataset}_{self.current_task}.pt')
            print(f"Predictions saved for task {self.current_task} in 'predictions_{self.args.dataset}_{self.current_task}.pt'")
            self.predictions = []
            self.original_labels = []
        return super().end_task(dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):

        self.opt.zero_grad()
        outputs = self.net(inputs, torch.unique(labels).tolist())
        loss = self.loss(outputs, (labels % 2))
        loss.backward()
        self.opt.step()

        return loss.item()

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]

