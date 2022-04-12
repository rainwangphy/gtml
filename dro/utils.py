import torch
from src.running_average import get_log_running_average, LogRunningAverage
import numpy as np
import scipy


def find_tau_star(ell, kappa, log_min=-10, log_max=10):
    # Find \tau^* such that KL(q^*_\tau^* || p) = \kappa
    # Heuristically we've found that values of \tau can be very small (<10^2)
    # or sometimes big (10^2). Therefore, searching for \log_10 \tau^* is
    # marginally faster since the values taken by \tau^* are more evenly
    # spread out on the log scale

    def kl_diff(log_tau):
        tau = 10 ** log_tau
        log_q_star_ = ell / tau
        log_q_star_ -= log_q_star_.max()
        log_q_star = log_q_star_ - np.log(np.mean(np.exp(log_q_star_)))
        return (np.exp(log_q_star) * log_q_star).mean() - kappa

    # Check boundary conditions
    if kl_diff(log_min) * kl_diff(log_max) >= 0:
        # In this case \tau^* lies outside the interval
        if np.abs(kl_diff(log_min)) < np.abs(kl_diff(log_max)):
            # \tau^* lies to the left of the interval so the minimum value
            # is our best guess
            log_tau_star = log_min
        else:
            # Or it lies to the right
            log_tau_star = log_max
    else:
        # \tau^* lies inside the interval, use the bisection method
        # (=binary search) to find it
        log_tau_star = scipy.optimize.bisect(
            kl_diff, log_min, log_max, xtol=1e-5, maxiter=100
        )

    return 10 ** (log_tau_star)


def compute_model_loss(
        losses: torch.Tensor,
        log_q: torch.Tensor,
        log_p: torch.Tensor,
        adv_args,
        log_Z_adv: LogRunningAverage,
        log_Z_model: LogRunningAverage,
        errors,
) -> torch.Tensor:
    """Computes the loss of the model given the model's los and the
    adversary's weights on each sample

    Args:
        losses: Loss of each sample (of shape [B])
        log_q: Log probability of each sample under the adversary
            (of shape [B])
        log_p: Log probability of each sample under the MLE model
            (of shape [B])
        adv_args: Arguments for the adversary
        log_Z_adv: Log normalizer for the adversary's weights
        log_Z_model: Log normalizer for the model's weights
        errors: some arbitrary error function of the model's output on each sample (of shape [B])

    Returns:
        Loss for the model on this batch (a scalar tensor)
    """
    # Compute the log ratio
    if adv_args.non_param:
        # Non-parametric adversary
        if adv_args.kappa is not None:
            # If the KL bound is fixed, we find the temperature which
            # satisfies it
            tau_star = find_tau_star(
                losses.detach().cpu().numpy(),
                adv_args.kappa,
            )
        else:
            # Otherwise just use a fixed temperature
            tau_star = adv_args.tau
        # Un-normalized q_star
        log_q_star_ = losses / tau_star
        # log normalize
        # Note that the log normalizer is
        # E_z~p e^{l(z)/\tau} which we approximate with the empirical average
        # of e^{l/\tau} over the minibatch
        log_Z = torch.logsumexp(log_q_star_ - np.log(len(losses)), dim=0)
        log_q_star = log_q_star_ - log_Z
        # Weights for the loss function
        # Notice that we don't detach tthe weights: we will backprop
        # through q_\tau,\theta too
        model_loss = (torch.exp(log_q_star) * losses).mean()
    else:
        log_ratios = log_q - log_p
        # Renormalize weights
        log_ratios = log_ratios - log_Z_model.value
        # Importance weights
        weights = torch.exp(log_ratios)
        # Loss
        model_loss = (weights.detach() * losses).sum()
    # Interpolate between the adversarial loss and the ERM objective
    # 1 means we are only training on the adversarial objective
    # 0 means we are only training on the ERM objective
    if adv_args.alpha < 1:
        erm_loss = losses.mean()
        model_loss = model_loss * adv_args.alpha + (1 - adv_args.alpha) * erm_loss

    return model_loss


def compute_adv_loss(
        losses: torch.Tensor,
        log_q: torch.Tensor,
        log_p: torch.Tensor,
        adv_args,
        log_Z_adv: LogRunningAverage,
        log_Z_model: LogRunningAverage,
        errors,
) -> torch.Tensor:
    """Compute the adversary's loss given the model's loss on a batch of
    examples and the weights produced by the adversary

    Args:
        losses: A tensor containing the losses of the model on a
            minibatch
        log_q: A tensor containing the probability of each example
            in the mininbatch
        log_p: A tensor containing the baseline probability for
            each example in the batch
        adv_args: Arguments specific to the adversary
        log_Z_adv: Running average of the weights used in
            computing the adversary's loss
        errors: Tensor containing the errors of the model on the
            minibatch (these can be non-differentiable, as opposed as the
            losses)
        log_Z_model: This is the log normalizer of the
            weights used to compute the model's loss. Here this is used to
            recompute the model loss in the `zero_sum` setting (where the
            adversary is trained to maximize the model's loss)
    """
    # Interpolate with the regular nll
    if adv_args.non_param:
        # Non parametric case: we don't train the adversary
        return torch.zeros(1, requires_grad=True)
    elif adv_args.adv_obj == "zero_sum":
        # LM NLL in log space:
        weights = (log_q - log_p) - log_Z_model.value
        adv_loss = -(torch.exp(weights) * losses.detach()).mean()
    elif adv_args.adv_obj == "fwd_kl":
        # Log likelihood ratio
        log_weights = (log_q - log_p) - log_Z_model.value
        # weights
        weights = torch.exp(log_weights)
        # "Entropy" component
        ent_loss = (weights * log_weights).mean()
        # "zero sum" component
        zero_sum_loss = (-weights * losses.detach()).mean()
        adv_loss = zero_sum_loss + adv_args.tau * ent_loss
    elif adv_args.adv_obj == "log_zero_sum":
        # LM NLL in log space:
        log_losses = log_q - log_p + torch.log(losses).detach()
        adv_loss = -torch.logsumexp(log_losses, 0)
    elif adv_args.adv_obj.startswith("exp"):
        if adv_args.adv_on_acc:
            log_q_star = errors / adv_args.tau
        else:
            # q*(x, y) \propto \ell(x, y)/temp * p
            log_q_star = losses.detach() / adv_args.tau
        if adv_args.adv_obj == "exp":
            # Reweight by log_p
            log_lm_weights = log_q_star - log_p
        elif adv_args.adv_obj == "exp_kl":
            # Reweight by log_p
            log_lm_weights = log_q_star
        # Actual weights are normalized across minibatch
        log_normalizer = torch.logsumexp(log_lm_weights, 0).item()
        # Running average
        log_Z_adv += log_normalizer
        # print(log_Z_adv.value, flush=True)
        # log_lm_weights += np.log(batch.size)
        lm_weights = torch.exp(log_lm_weights - log_Z_adv.value)
        # Loss for the lm
        adv_loss = -(lm_weights * log_q).sum()
    # # lm_loss = -(torch.exp(log_q-log_p)*nlls.detach()).mean()
    if adv_args.ratio_model and adv_args.self_norm_lambda > 0:
        log_expected_ratio = torch.logsumexp(log_q - np.log(len(log_q)), dim=0)
        adv_loss = adv_loss + adv_args.self_norm_lambda * log_expected_ratio ** 2
    # Interpolate with the likelihood of the data
    # (this pulls back the adversary towards the nominal distribution)
    if adv_args.beta < 1:
        adv_loss = adv_args.beta * adv_loss + (1 - adv_args.beta) * (-log_q).mean()
    return adv_loss
