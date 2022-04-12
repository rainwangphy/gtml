import numpy as np
import torch
import torch.nn.functional as F

from src.data.language_modeling import to_lm_batch
from src.models import build_model, ModelWithHead
from src.optim import get_optimizer, get_lr_scheduler
from src.running_average import get_log_running_average
import dro.utils as utils


class p_dro_model:
    def __int__(self, args):
        self.args = args
        self.task = args.task

        self.model = self.get_model()
        self.m_opt = self.get_opt()
        self.lr_scheduler = self.get_lr_scheduler()

        self.current_step = 0

    def train(self, data, adversary):
        self.model.train()
        adversary.eval()
        self.current_step += 1
        if (self.current_step - 1) % self.args.model_args.update_every == 0:
            self.m_opt.zero_grad()
        model_loss = self.forward(data, adversary)

        if self.args.model_args.l2_reg > 0:
            param_vec = torch.cat([p.view(-1) for p in self.model.parameters()])
            model_loss = model_loss + self.args.model_args.l2_reg * torch.sum(
                param_vec ** 2
            )

        model_loss.backward()
        if self.current_step % self.args.model_args.update_every == 0:
            # Clip model gradient
            if self.args.model_args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.model_args.clip_grad,
                )
            # Update params and LR
            self.m_opt.step()
            self.lr_scheduler.step()

    def eval(self, data, adversary):
        self.model.eval()
        adversary.eval()
        model_loss = self.forward(data, adversary)
        return model_loss.detach().cpu().numpy()

    def forward(self, data, adversary):
        task = self.task
        model = self.model
        batch = data
        adv_args = self.args.adv_args
        adv = adversary.adversary
        adv_task = adv.task
        nlls, _, y_hat = task.nll(
            model,
            batch,
            reduction="none",
            predict=True,
        )
        # Model errors
        errors = (batch.outputs != y_hat).float().detach()
        log_Z_model = get_log_running_average(adv_args.norm_k_model)
        log_Z_adv = get_log_running_average(adv_args.norm_k_adv)
        # Transform the minibatch for processing by the adversary
        lm_batch = batch
        if not (adv_args.joint or adv_args.class_conditional):
            lm_batch = to_lm_batch(lm_batch)
        # Get log prob of each sample under the adversary
        if adv_args.ratio_model:
            logits = adv_task.logits(adv, batch)
            y = batch.outputs.to(logits.device)
            log_q = -F.nll_loss(logits, y, reduction="none")
            if adv_args.renorm_ratios:
                log_q = torch.log_softmax(log_q, dim=0) + np.log(len(log_q))
        else:
            # Get NLL for words
            log_q = -adv_task.nll(adv, lm_batch, reduction="none")
            # Sum along the length dimention
            log_q = log_q.sum(-1)
        # log prob under the MLE LM
        log_p = torch.tensor(batch.attributes["log_p"]).to(log_q.device)
        # Keep track of the log normalizer for the weights used to
        # compute the model's loss
        log_Z_model += torch.logsumexp(log_q - log_p, 0).item()
        model_loss = utils.compute_model_loss(
            nlls, log_q, log_p, adv_args, log_Z_adv, log_Z_model, errors
        )

        return model_loss

    def get_model(self):
        # print()
        model_args = self.args.model_args
        input_shape = self.args.input_shape
        model = build_model(model_args.architecture, input_shape, None)
        head = self.task.create_compatible_head(model.hidden_size)
        model = ModelWithHead(model, head)
        return model

    def get_opt(self):
        return get_optimizer(
            self.args.optimizer,
            list(self.model.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def get_lr_scheduler(self):
        args = self.args
        return get_lr_scheduler(
            args.lr_scheduler,
            self.m_opt,
            args.lr,
            args.n_steps,
        )


class p_dro_adversary:
    def __init__(self, args):
        self.args = args
        self.adversary = self.get_adversary()
        self.adv_opt = self.get_opt()

        self.adv_tasks = args.adv_tasks

        self.current_step = 0

    def train(self, data, model):
        print()
        model.eval()
        self.adversary.eval()
        self.current_step += 1
        adv_args = self.args.adv_args
        self.adv_opt.zero_grad()
        adv_loss = self.forward(data, model)
        adv_loss.backward()
        if self.current_step % adv_args.adv_update_every == 0:
            # Clip adv gradient
            if adv_args.clip_grad_adv > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.adversary.parameters(),
                    adv_args.clip_grad_adv,
                )
            # Update adversary
            self.adv_opt.step()

    def eval(self, data, model):
        adv_loss = self.forward(data, model)
        return adv_loss.detach().cpu().numpy()

    def forward(self, data, model):
        # print()

        task = model.task
        model = model.model
        batch = data
        adv_args = self.args.adv_args
        adv = self.adversary
        adv_task = adv.task
        nlls, _, y_hat = task.nll(
            model,
            batch,
            reduction="none",
            predict=True,
        )
        # Model errors
        errors = (batch.outputs != y_hat).float().detach()
        log_Z_model = get_log_running_average(adv_args.norm_k_model)
        log_Z_adv = get_log_running_average(adv_args.norm_k_adv)
        # Transform the minibatch for processing by the adversary
        lm_batch = batch
        if not (adv_args.joint or adv_args.class_conditional):
            lm_batch = to_lm_batch(lm_batch)
        # Get log prob of each sample under the adversary
        if adv_args.ratio_model:
            logits = adv_task.logits(adv, batch)
            y = batch.outputs.to(logits.device)
            log_q = -F.nll_loss(logits, y, reduction="none")
            if adv_args.renorm_ratios:
                log_q = torch.log_softmax(log_q, dim=0) + np.log(len(log_q))
        else:
            # Get NLL for words
            log_q = -adv_task.nll(adv, lm_batch, reduction="none")
            # Sum along the length dimention
            log_q = log_q.sum(-1)
        # log prob under the MLE LM
        log_p = torch.tensor(batch.attributes["log_p"]).to(log_q.device)
        # Keep track of the log normalizer for the weights used to
        # compute the model's loss
        log_Z_model += torch.logsumexp(log_q - log_p, 0).item()
        # model_loss = utils.compute_model_loss(
        #     nlls, log_q, log_p, adv_args, log_Z_adv, log_Z_model, errors
        # )
        # Compute the adversary's loss
        adv_loss = utils.compute_adv_loss(
            nlls, log_q, log_p, adv_args, log_Z_adv, log_Z_model, errors
        )
        return adv_loss

    def get_adversary(self):
        # print()
        architecture = self.args.adv_architecture
        ratio_model = self.args.ratio_model
        filename = self.args.adv_filename
        input_shape = self.args.input_shape
        output_size = self.args.adv_output_size
        device = self.args.device
        """Create the adversary

        Args:
            architecture (str): Architecture
            filename (str): Path to MLE model
            input_shape (Tuple[int, ...]): Shape of the inputs.
            output_size (Tuple[int, ...]): Shape of the outputs.
            device (str, optional): Device. Defaults to "gpu:1".
            ratio_model (bool, optional): TODO. Defaults to False.

        Returns:
            The adversary with the MLE parameters loaded in
        """
        if ratio_model:
            # In this case, the adversary models the ratio q / p directly.
            # It is a model that takes an input and returns a real number
            # log (q / p), unnormalized. If we are modeling the join distribution,
            # This returns a vector where row i corresponds
            # to log (q(x, i)/p(x, i))
            adv = build_model(architecture, input_shape, output_size=None)
            head = torch.nn.Linear(adv.hidden_size, output_size)
            # We initialize the head at 0, so the starting log ratio will be 0
            # But this is not necessary after all
            # head.weight.data.zero_()
            # head.bias.data.zero_()
            adv = ModelWithHead(adv, head)
        else:
            adv = build_model(architecture, input_shape, output_size)
            # There is no classification head
            adv = ModelWithHead(adv)
        # Maybe load a pre-trained model
        if filename is not None:
            adv_state_dict = torch.load(filename, map_location=device)
            adv.load_state_dict(adv_state_dict, strict=False)
        return adv.to(device)

    def get_opt(self):
        # Optimizer for the adversary
        # Default to the model's optimizer
        adv_args = self.args.adv_args
        adv_optimizer = adv_args.adv_optimizer
        # if adv_optimizer is None:
        #     adv_optimizer = optim_args.optimizer
        return get_optimizer(
            adv_optimizer,
            list(self.adversary.parameters()),
            lr=adv_args.adv_lr,
            mom=adv_args.adv_mom,
            weight_decay=adv_args.weight_decay,
        )
