import numpy as np
import torch
import torch.nn.functional as F

from src.data.language_modeling import to_lm_batch
from src.models import build_model, ModelWithHead
from src.optim import get_optimizer, get_lr_scheduler
from src.running_average import get_log_running_average
import dro.utils as utils


class p_dro_model:
    def __init__(self, config):
        self.config = config
        # print(config)
        self.task = config.model_config.task

        self.model = self.get_model()
        self.m_opt = self.get_opt()
        self.lr_scheduler = self.get_lr_scheduler()

        self.current_step = 0

    def train(self, data, adversary):
        self.model.to(self.config.device)
        adversary.adversary.to(self.config.device)
        data = data.to(self.config.device)
        self.model.train()
        adversary.adversary.eval()
        self.current_step += 1
        if (self.current_step - 1) % self.config.model_config.update_every == 0:
            self.m_opt.zero_grad()
        model_loss = self.forward(data, adversary)

        if self.config.model_config.l2_reg > 0:
            param_vec = torch.cat([p.view(-1) for p in self.model.parameters()])
            model_loss = model_loss + self.config.model_config.l2_reg * torch.sum(
                param_vec ** 2
            )

        model_loss.backward()
        if self.current_step % self.config.model_config.update_every == 0:
            # Clip model gradient
            if self.config.model_config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.model_config.clip_grad,
                )
            # Update params and LR
            self.m_opt.step()
            self.lr_scheduler.step()

    def eval(self, data, adversary):
        self.model.to(self.config.device)
        adversary.adversary.to(self.config.device)
        data = data.to(self.config.device)
        self.model.eval()
        adversary.adversary.eval()
        # adversary.eval()
        model_loss = self.forward(data, adversary)
        return model_loss.detach().cpu().numpy()

    def forward(self, data, adversary):
        task = self.task
        model = self.model
        batch = data
        adv_config = self.config.adv_config
        adv = adversary.adversary
        adv_task = adversary.adv_task
        nlls, _, y_hat = task.nll(
            model,
            batch,
            reduction="none",
            predict=True,
        )
        # Model errors
        errors = (batch.outputs != y_hat).float().detach()
        log_Z_model = get_log_running_average(adv_config.norm_k_model)
        log_Z_adv = get_log_running_average(adv_config.norm_k_adv)
        # Transform the minibatch for processing by the adversary
        lm_batch = batch
        if not (adv_config.joint or adv_config.class_conditional):
            lm_batch = to_lm_batch(lm_batch)
        # Get log prob of each sample under the adversary
        if adv_config.ratio_model:
            logits = adv_task.logits(adv, batch)
            y = batch.outputs.to(logits.device)
            log_q = -F.nll_loss(logits, y, reduction="none")
            if adv_config.renorm_ratios:
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
            nlls, log_q, log_p, adv_config, log_Z_adv, log_Z_model, errors
        )

        return model_loss

    def get_model(self):
        # print()
        model_config = self.config.model_config
        input_shape = model_config.input_shape
        model = build_model(model_config.architecture, input_shape, None)
        head = self.task.create_compatible_head(model.hidden_size)
        model = ModelWithHead(model, head)
        return model

    def get_opt(self):
        return get_optimizer(
            self.config.model_config.optimizer,
            list(self.model.parameters()),
            lr=self.config.model_config.lr,
            weight_decay=self.config.model_config.weight_decay,
        )

    def get_lr_scheduler(self):
        config = self.config
        return get_lr_scheduler(
            config.model_config.lr_scheduler,
            self.m_opt,
            config.model_config.lr,
            config.model_config.n_steps,
        )


class p_dro_adversary:
    def __init__(self, config):
        self.config = config
        # print(config)
        self.adversary = self.get_adversary()
        self.adv_opt = self.get_opt()

        self.adv_task = config.adv_config.adv_task

        self.current_step = 0

    def train(self, data, model):
        model.model.to(self.config.device)
        self.adversary.to(self.config.device)
        data = data.to(self.config.device)
        model.model.eval()
        self.adversary.eval()
        self.current_step += 1
        adv_config = self.config.adv_config
        self.adv_opt.zero_grad()
        adv_loss = self.forward(data, model)
        adv_loss.backward()
        if self.current_step % adv_config.adv_update_every == 0:
            # Clip adv gradient
            if adv_config.clip_grad_adv > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.adversary.parameters(),
                    adv_config.clip_grad_adv,
                )
            # Update adversary
            self.adv_opt.step()

    def eval(self, data, model):
        model.model.to(self.config.device)
        self.adversary.to(self.config.device)
        data = data.to(self.config.device)
        adv_loss = self.forward(data, model)
        return adv_loss.detach().cpu().numpy()

    def forward(self, data, model):
        task = model.task
        model = model.model
        batch = data
        adv_config = self.config.adv_config
        adv = self.adversary
        adv_task = self.adv_task
        nlls, _, y_hat = task.nll(
            model,
            batch,
            reduction="none",
            predict=True,
        )
        # Model errors
        errors = (batch.outputs != y_hat).float().detach()
        log_Z_model = get_log_running_average(adv_config.norm_k_model)
        log_Z_adv = get_log_running_average(adv_config.norm_k_adv)
        # Transform the minibatch for processing by the adversary
        lm_batch = batch
        if not (adv_config.joint or adv_config.class_conditional):
            lm_batch = to_lm_batch(lm_batch)
        # Get log prob of each sample under the adversary
        if adv_config.ratio_model:
            logits = adv_task.logits(adv, batch)
            y = batch.outputs.to(logits.device)
            log_q = -F.nll_loss(logits, y, reduction="none")
            if adv_config.renorm_ratios:
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
        #     nlls, log_q, log_p, adv_config, log_Z_adv, log_Z_model, errors
        # )
        # Compute the adversary's loss
        adv_loss = utils.compute_adv_loss(
            nlls, log_q, log_p, adv_config, log_Z_adv, log_Z_model, errors
        )
        return adv_loss

    def get_adversary(self):
        """Create the adversary

        config:
            architecture (str): Architecture
            filename (str): Path to MLE model
            input_shape (Tuple[int, ...]): Shape of the inputs.
            output_size (Tuple[int, ...]): Shape of the outputs.
            device (str, optional): Device. Defaults to "gpu:1".
            ratio_model (bool, optional): TODO. Defaults to False.

        Returns:
            The adversary with the MLE parameters loaded in
        """
        architecture = self.config.adv_config.adv_architecture
        ratio_model = self.config.adv_config.ratio_model
        filename = self.config.adv_config.adv_filename
        input_shape = self.config.adv_config.input_shape
        output_size = self.config.adv_config.adv_output_size
        device = self.config.device
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
        adv_config = self.config.adv_config
        adv_optimizer = adv_config.adv_optimizer
        # if adv_optimizer is None:
        #     adv_optimizer = optim_config.optimizer
        return get_optimizer(
            adv_optimizer,
            list(self.adversary.parameters()),
            lr=adv_config.adv_lr,
            mom=adv_config.adv_mom,
            weight_decay=adv_config.weight_decay,
        )


# config = {
#     "def_lala": 20
# }
#
# p_dro_model_stance = p_dro_proto(config)
# p_dro_adversary_stance = p_dro_adversary(config=config)

# print(p_dro_model_stance)
