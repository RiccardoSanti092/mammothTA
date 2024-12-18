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
from torch import func
import torch.optim as optim



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


class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_embedding.type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class FinalModel(nn.Module):
    def __init__(self, clip_model, dataset: ContinualDataset, args) -> None:
        super().__init__()
        self.dataset = dataset
        self.clip_model = clip_model

        #self.clip_model.eval()
        self.args = args

        self.classes = self.dataset.get_class_names()
        if args.use_templates:
            templates = self.dataset.get_prompt_templates()
            text_inputs = []
            for t in templates:
                t_inputs = torch.cat([clip.tokenize(t.format(c)) for c in self.classes]).to(get_device())
                t_inputs = self.clip_model.encode_text(t_inputs)
                t_inputs /= t_inputs.norm(dim=-1, keepdim=True)  # double normalization if use templates is expected (see https://github.dev/KaiyangZhou/CoOp)
                text_inputs.append(t_inputs)
            self.text_features = torch.stack(text_inputs).mean(0)
        else:
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.classes]).to(get_device())
            self.text_features = self.clip_model.encode_text(text_inputs)

        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)  # double normalization if use templates is expected
        self.task_id = 0


    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        text_features = self.text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * (image_features @ text_features.T))

        return similarity


class CLIP(ContinualModel):
    """STATIC Continual Learning with CLIP"""
    NAME = 'clipTA'
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
        backbone, clip_transform = clip.load(args.clip_backbone, device=get_device())
        n_epochs = 1 if args.save_predictions else 0
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.net = FinalModel(self.net, self.dataset, args)

        #for param in self.net.parameters():
        #    param.requires_grad = False
        #for param in self.net.clip_model.visual.parameters():
        #    param.requires_grad = True
        #self.net.clip_model.visual.training = True

        for param in self.net.parameters():
            param.requires_grad = True


        #self.net_w0 = deepcopy(self.net)
        #self.final_net = deepcopy(self.net.eval())
        #set_params(self.final_net, torch.zeros_like(get_params(self.final_net)))  #ROMPE
        #self.delta_w_params = nn.ParameterList(self.final_net.parameters())

        #for param in self.delta_w_params:
        #    param.data = torch.zeros_like(param.data)
        #    param.requires_grad = False #TODO: imposta a True quando linearizzi


        self.clip_transform = clip_transform

        self.predictions = []
        self.original_labels = []

    def begin_task(self, dataset):


        dataset.test_loaders[-1].dataset.transform = self.clip_transform
        if self.args.save_predictions:
            dataset.train_loader.dataset.transform = self.clip_transform

        if self.current_task != 0:
            self.net.task_id += 1



    def end_task(self, dataset: ContinualDataset) -> None:
        if self.args.save_predictions:
            self.predictions = torch.cat(self.predictions, dim=0).cpu()
            self.original_labels = torch.cat(self.original_labels, dim=0).cpu()
            torch.save((self.predictions, self.original_labels), f'predictions_{self.args.dataset}_{self.current_task}.pt')
            print(f"Predictions saved for task {self.current_task} in 'predictions_{self.args.dataset}_{self.current_task}.pt'")
            self.predictions = []
            self.original_labels = []
        return super().end_task(dataset)


    def begin_epoch(self, epoch: int, dataset: ContinualDataset) -> None:
        torch.cuda.empty_cache()


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        similarity = self.net(inputs.to(dtype=self.net.clip_model.dtype))
        loss_xe = nn.CrossEntropyLoss()
        loss = loss_xe(similarity, labels)
        return loss


        #features, jvp = func.jvp(func_network, (tuple(self.net_w0),), (tuple(self.delta_w_params),),# devo capire se deltaW é bene farlo solo sulla parte visiva oppure su tutto)






    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]
