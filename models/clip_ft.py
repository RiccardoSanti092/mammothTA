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
from torch import func
from typing import Iterable, Tuple
from tqdm import tqdm
import gc


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
        parser.add_argument('--ft_linears', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune linear layers')
        parser.add_argument('--ft_attention', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune attention layers')
        parser.add_argument('--ft_ln', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune layer norm')
        parser.add_argument('--ft_class_embed', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune class embedding layers')
        parser.add_argument('--ft_pos_embed', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune posistional embedding')
        parser.add_argument('--ft_proj', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune projection layers')
        parser.add_argument('--ft_conv', type=binary_to_boolean_type, default=0, help='Set to 1 fine-tune convolutional layers')

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

    def begin_task(self, dataset):
        torch.cuda.empty_cache()
        dataset.test_loaders[-1].dataset.transform = self.clip_transform
        dataset.train_loader.dataset.transform = self.clip_transform

        if self.current_task != 0:
            self.net.task_id += 1

        self.delta_w = []
        for name, param in self.net.visual_encoder.named_parameters():
            if self.args.ft_linears and "mlp" in name:
                param.requires_grad = True
            elif self.args.ft_attention and "attn" in name:
                param.requires_grad = True
            elif self.args.ft_class_embed and "class_embed" in name:
                param.requires_grad = True
            elif self.args.ft_conv and "conv" in name:
                param.requires_grad = True
            elif self.args.ft_ln and "ln" in name:
                param.requires_grad = True
            elif self.args.ft_pos_embed and "positional_embedding" in name:
                param.requires_grad = True
            elif self.args.ft_proj and "proj" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            self.delta_w.append(param)



        self.opt = optim.SGD(self.delta_w, lr=self.args.lr,
                             momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)

        self.train()

    def end_task(self, dataset: ContinualDataset) -> None:
        print("Current task:")
        print(self.current_task)
        if self.args.save_predictions:
            self.predictions = torch.cat(self.predictions, dim=0).cpu()
            self.original_labels = torch.cat(self.original_labels, dim=0).cpu()
            torch.save((self.predictions, self.original_labels), f'predictions_{self.args.dataset}_{self.current_task}.pt')
            print(f"Predictions saved for task {self.current_task} in 'predictions_{self.args.dataset}_{self.current_task}.pt'")
            self.predictions = []
            self.original_labels = []

        task_vector_dict = {name: param_finetuned - param_pretrained
                            for ((name, param_pretrained), (param_finetuned))
                            in zip(self.net.named_parameters(), self.delta_w)}

        #torch.save(task_vector_dict, f"C:\Riccardo\Dottorato\CGIL Variance Collapse\TASK VECTORS\\task_vector{self.current_task}.pt")



        #TODO: check why task vectors remain on CUDA gaurdando tmc_peft

        ##########################################################################################
        ##########################################################################################
        '''
        if self.current_task > 0:
            task = random.randint(0, self.current_task)
            print(f"Testing task {task}")
            self.test_task_vector = deepcopy(self.task_vector_list[task])
            for name, param in self.net.named_parameters():
                if name in self.test_task_vector:
                    self.test_task_vector[name] += param.data
                else:
                    self.test_task_vector[name] = param.data.clone()


            print("Evaluation con task singolo casuale.\n")
            accs = self.check_correctnes(dataset)
            print(accs)
        

            del self.test_task_vector
        '''
        ##########################################################################################
        ##########################################################################################


        if self.current_task > 0:
            for key in self.merged_params:
                self.merged_params[key] *= self.current_task
                self.merged_params[key] += task_vector_dict[key]
                self.merged_params[key] /= (self.current_task) + 1
            print("Media parametri aggiornata.")
        else:
            self.merged_params = task_vector_dict
            print("Media parametri aggiornata.")

        task_vector_dict = None
        torch.no_grad()
        self.opt = None
        del task_vector_dict, self.opt, self.delta_w
        gc.collect()
        '''
        self.merged_parames = {}
        for task_vector in self.task_vector_list:
            for key, tensor in task_vector.items():
                if key in self.merged_parames:
                    self.merged_parames[key] += tensor
                else:
                    self.merged_parames[key] = tensor.clone()

        #TODO: implementa il merging di Peet che é piú elegante a scrittura

        if self.current_task > 0:
            print("Averaging task vectors")
            for key, tensor in self.merged_parames.items():
                tensor /= (self.current_task + 1)
        '''
        self.eval_params = deepcopy(self.net)
        for name, param in self.eval_params.named_parameters():
            if name in self.merged_params:
                param = param + self.merged_params[name]  # TODO param.data

        torch.cuda.empty_cache()
        return super().end_task(dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch = None):

        self.opt.zero_grad()
        param = {name: param for name, param in zip(self.param_names, self.delta_w)}
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
    def forward(self, x, correctness: bool = False):#TODO: passa una booleana che in base a quello usa i parametri giusti
        #param = {name: param for name, param in zip(self.param_names, self.params_optimizer)}

        image_features = func.functional_call(self.net,  {name: param for name, param in self.eval_params.named_parameters()}, x)
        similarity = (100.0 * (image_features @ self.net.text_features.T)).softmax(dim=-1)
        return similarity[:, :self.n_seen_classes]

    @torch.no_grad()
    def forward_old(self, x):
        return self.net(x)[:, :self.n_seen_classes]








    def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
        """
        Given the output tensor, the dataset at hand and the current task,
        masks the former by setting the responses for the other tasks at -inf.
        It is used to obtain the results for the task-il setting.

        Args:
            outputs: the output tensor
            dataset: the continual dataset
            k: the task index
        """
        num_classes = dataset.N_CLASSES
        start_c, end_c = dataset.get_offsets(k)
        outputs[:, :start_c] = -float('inf')
        outputs[:, end_c:num_classes] = -float('inf')


    def check_correctnes(self, dataset: ContinualDataset, last=False, return_loss=False) -> Tuple[list, list]:

        status = self.net.training
        self.net.eval()
        accs, accs_mask_classes = [], []
        n_classes = dataset.get_offsets()[1]
        loss_fn = dataset.get_loss()
        avg_loss = 0
        total_len = sum(len(x) for x in dataset.test_loaders) if hasattr(dataset.test_loaders[0], '__len__') else None

        pbar = tqdm(dataset.test_loaders, total=total_len, desc='Evaluating', disable=self.args.non_verbose)
        for k, test_loader in enumerate(dataset.test_loaders):
            if last and k < len(dataset.test_loaders) - 1:
                continue
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            test_iter = iter(test_loader)
            i = 0
            while True:
                try:
                    data = next(test_iter)
                except StopIteration:
                    break
                if self.args.debug_mode and i > self.get_debug_iters():
                    break
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)

                if return_loss:
                    loss = loss_fn(outputs, labels)
                    avg_loss += loss.item()

                _, pred = torch.max(outputs[:, :n_classes].data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                i += 1
                pbar.set_postfix({f'acc_task_{k + 1}': max(0, correct / total * 100)}, refresh=False)
                pbar.set_description(f"Evaluating Task {k + 1}", refresh=False)
                pbar.update(1)

                if dataset.SETTING == 'class-il':
                    self.mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

            accs.append(correct / total * 100
                        if 'class-il' in self.COMPATIBILITY or 'general-continual' in self.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)
        pbar.close()

        self.net.train(status)
        if return_loss:
            return accs, accs_mask_classes, avg_loss / total
        return accs, accs_mask_classes