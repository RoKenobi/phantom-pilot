import numpy as np
import pytest
import torch

from agents.networks import ActorCritic, ObsSplitter, QNetwork, ScanEncoder, layer_init


def test_layer_init_sets_orthogonal_weights_and_constant_bias():
    layer = layer_init(torch.nn.Linear(8, 8), std=1.0, bias_const=0.5)
    w = layer.weight.detach()
    assert torch.allclose(w @ w.T, torch.eye(8), atol=1e-5)
    assert torch.allclose(layer.bias.detach(), torch.full((8,), 0.5))


def test_scan_encoder_output_shape():
    enc = ScanEncoder(n_rays=24, in_channels=1, out_dim=64)
    out = enc(torch.zeros(5, 1, 24))
    assert out.shape == (5, 64)


def test_scan_encoder_accepts_stacked_frames_as_channels():
    enc = ScanEncoder(n_rays=24, in_channels=4, out_dim=64)
    assert enc(torch.zeros(3, 4, 24)).shape == (3, 64)


def test_scan_encoder_uses_replicate_padding_by_default():
    """Circular padding would be wrong for a 270-degree FOV."""
    enc = ScanEncoder(n_rays=24)
    modes = [m.padding_mode for m in enc.modules() if isinstance(m, torch.nn.Conv1d)]
    assert modes and all(m == "replicate" for m in modes)


def test_scan_encoder_is_translation_sensitive_but_not_degenerate():
    enc = ScanEncoder(n_rays=24, in_channels=1, out_dim=32)
    a = enc(torch.zeros(1, 1, 24))
    b = enc(torch.ones(1, 1, 24))
    assert not torch.allclose(a, b)


def test_obs_splitter_single_frame():
    sp = ObsSplitter(n_rays=24, n_extra=5, frame_stack=1)
    obs = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    scan, extra = sp.split(obs)
    assert scan.shape == (1, 1, 24) and extra.shape == (1, 5)
    assert torch.allclose(scan[0, 0], torch.arange(24, dtype=torch.float32))
    assert torch.allclose(extra[0], torch.arange(24, 29, dtype=torch.float32))


def test_obs_splitter_stacked_frames_keep_frame_order():
    sp = ObsSplitter(n_rays=24, n_extra=5, frame_stack=3)
    obs = torch.arange(3 * 29, dtype=torch.float32).unsqueeze(0)
    scan, extra = sp.split(obs)
    assert scan.shape == (1, 3, 24) and extra.shape == (1, 15)
    assert scan[0, 0, 0] == 0.0
    assert scan[0, 1, 0] == 29.0, "frame 1's scan starts at index 29"


def test_actor_critic_output_shapes():
    net = ActorCritic(n_rays=24, n_extra=5, frame_stack=1, act_dim=2)
    mean, value = net(torch.zeros(7, 29))
    assert mean.shape == (7, 2) and value.shape == (7,)


def test_actor_critic_has_learnable_log_std_of_correct_size():
    net = ActorCritic(n_rays=24, n_extra=5, frame_stack=1, act_dim=2)
    assert isinstance(net.log_std, torch.nn.Parameter)
    assert net.log_std.shape == (2,)
    assert net.log_std.requires_grad


def test_actor_critic_policy_head_starts_near_zero():
    """Small final-layer gain keeps the initial policy close to the action
    space centre -- a standard PPO implementation detail."""
    net = ActorCritic(n_rays=24, n_extra=5, frame_stack=1, act_dim=2)
    mean, _ = net(torch.randn(64, 29) * 0.1)
    assert mean.abs().mean().item() < 0.2


def test_actor_critic_works_with_frame_stacking():
    net = ActorCritic(n_rays=24, n_extra=5, frame_stack=4, act_dim=2)
    mean, value = net(torch.zeros(2, 4 * 29))
    assert mean.shape == (2, 2) and value.shape == (2,)


def test_actor_critic_gradients_flow_to_all_parameters():
    net = ActorCritic(n_rays=24, n_extra=5, frame_stack=1, act_dim=2)
    mean, value = net(torch.randn(8, 29))
    (mean.sum() + value.sum() + net.log_std.sum()).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"


def test_q_network_output_shape():
    net = QNetwork(n_rays=24, n_extra=5, frame_stack=1, n_actions=6)
    assert net(torch.zeros(9, 29)).shape == (9, 6)


def test_q_network_works_with_frame_stacking():
    net = QNetwork(n_rays=24, n_extra=5, frame_stack=4, n_actions=6)
    assert net(torch.zeros(2, 4 * 29)).shape == (2, 6)


def test_networks_are_small_enough_for_cpu():
    ac = ActorCritic(n_rays=24, n_extra=5, frame_stack=4, act_dim=2)
    n = sum(p.numel() for p in ac.parameters())
    assert n < 500_000, f"{n} parameters is too heavy for CPU-only training"
