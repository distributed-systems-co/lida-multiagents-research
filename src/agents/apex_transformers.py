"""APEX Transformers - Neural Prompt Manipulation.

Advanced neural architectures for prompt evolution:
1. Attention-based prompt encoding
2. Graph attention for structural relationships
3. Variational prompt autoencoders
4. Diffusion-based prompt generation
5. Reinforcement learning for prompt optimization
6. Meta-learning for few-shot adaptation
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np


# =============================================================================
# Multi-Head Self-Attention
# =============================================================================


class MultiHeadAttention:
    """Multi-head self-attention for prompt sequences.

    Implements scaled dot-product attention with multiple heads.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        # Initialize weights
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = np.random.randn(n_heads, d_model, self.d_k) * scale
        self.W_k = np.random.randn(n_heads, d_model, self.d_k) * scale
        self.W_v = np.random.randn(n_heads, d_model, self.d_k) * scale
        self.W_o = np.random.randn(n_heads * self.d_k, d_model) * scale

    def _scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scaled dot-product attention."""
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-1, -2) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        attn_weights = self._softmax(scores)

        # Apply dropout during training
        if self.dropout > 0:
            mask = np.random.rand(*attn_weights.shape) > self.dropout
            attn_weights = attn_weights * mask / (1 - self.dropout)

        return attn_weights @ V, attn_weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            x: Input tensor of shape (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Attended tensor of shape (seq_len, d_model)
            attention: Attention weights
        """
        seq_len = x.shape[0]
        all_heads = []
        all_attns = []

        for h in range(self.n_heads):
            Q = x @ self.W_q[h]
            K = x @ self.W_k[h]
            V = x @ self.W_v[h]

            head_out, attn = self._scaled_dot_product_attention(Q, K, V, mask)
            all_heads.append(head_out)
            all_attns.append(attn)

        # Concatenate heads
        concat = np.concatenate(all_heads, axis=-1)
        output = concat @ self.W_o

        # Average attention across heads for interpretability
        avg_attn = np.mean(all_attns, axis=0)

        return output, avg_attn


# =============================================================================
# Graph Attention Network for Prompt Structure
# =============================================================================


class GraphAttentionLayer:
    """Graph Attention Layer for prompt structure.

    Learns relationships between prompt components.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        alpha: float = 0.2,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha = alpha

        # Per-head parameters
        self.W = np.random.randn(n_heads, in_features, out_features) * 0.1
        self.a = np.random.randn(n_heads, 2 * out_features) * 0.1

    def _leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def forward(
        self,
        x: np.ndarray,
        adj: np.ndarray,
    ) -> np.ndarray:
        """Forward pass.

        Args:
            x: Node features (n_nodes, in_features)
            adj: Adjacency matrix (n_nodes, n_nodes)

        Returns:
            Updated node features (n_nodes, n_heads * out_features)
        """
        n_nodes = x.shape[0]
        outputs = []

        for h in range(self.n_heads):
            # Linear transformation
            Wh = x @ self.W[h]  # (n_nodes, out_features)

            # Attention coefficients
            a_input = np.concatenate([
                np.tile(Wh[:, np.newaxis, :], (1, n_nodes, 1)),
                np.tile(Wh[np.newaxis, :, :], (n_nodes, 1, 1)),
            ], axis=-1)

            e = self._leaky_relu(a_input @ self.a[h])

            # Mask with adjacency
            e = np.where(adj > 0, e, -1e9)

            # Softmax
            attention = np.exp(e - np.max(e, axis=1, keepdims=True))
            attention = attention / (np.sum(attention, axis=1, keepdims=True) + 1e-10)

            # Apply attention
            h_prime = attention @ Wh
            outputs.append(h_prime)

        return np.concatenate(outputs, axis=-1)


class PromptGraphAttention:
    """Graph Attention Network for prompt structure understanding."""

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        n_heads: int = 4,
    ):
        self.layer1 = GraphAttentionLayer(feature_dim, hidden_dim, n_heads)
        self.layer2 = GraphAttentionLayer(hidden_dim * n_heads, hidden_dim, n_heads)

    def build_adjacency(self, n_components: int) -> np.ndarray:
        """Build adjacency matrix for prompt components.

        Components: identity -> capabilities -> constraints -> instructions
        """
        adj = np.eye(n_components)

        # Sequential connections
        for i in range(n_components - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1

        # Identity connects to all
        adj[0, :] = 1
        adj[:, 0] = 1

        return adj

    def forward(
        self,
        component_features: np.ndarray,
    ) -> np.ndarray:
        """Process prompt components through GAT."""
        n_components = component_features.shape[0]
        adj = self.build_adjacency(n_components)

        h = self.layer1.forward(component_features, adj)
        h = np.maximum(h, 0)  # ReLU
        h = self.layer2.forward(h, adj)

        return h


# =============================================================================
# Variational Prompt Autoencoder
# =============================================================================


class PromptVAE:
    """Variational Autoencoder for prompt latent space.

    Learns a smooth latent space for prompt interpolation and generation.
    """

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Encoder
        dims = [input_dim] + hidden_dims
        self.enc_weights = []
        self.enc_biases = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.enc_weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.enc_biases.append(np.zeros(dims[i+1]))

        # Mean and log-variance
        scale = np.sqrt(2.0 / hidden_dims[-1])
        self.mu_weight = np.random.randn(hidden_dims[-1], latent_dim) * scale
        self.mu_bias = np.zeros(latent_dim)
        self.logvar_weight = np.random.randn(hidden_dims[-1], latent_dim) * scale
        self.logvar_bias = np.zeros(latent_dim)

        # Decoder
        dec_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.dec_weights = []
        self.dec_biases = []
        for i in range(len(dec_dims) - 1):
            scale = np.sqrt(2.0 / dec_dims[i])
            self.dec_weights.append(np.random.randn(dec_dims[i], dec_dims[i+1]) * scale)
            self.dec_biases.append(np.zeros(dec_dims[i+1]))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input to latent distribution parameters."""
        h = x
        for w, b in zip(self.enc_weights, self.enc_biases):
            h = self._relu(h @ w + b)

        mu = h @ self.mu_weight + self.mu_bias
        logvar = h @ self.logvar_weight + self.logvar_bias

        return mu, logvar

    def reparameterize(
        self,
        mu: np.ndarray,
        logvar: np.ndarray,
    ) -> np.ndarray:
        """Reparameterization trick."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to output."""
        h = z
        for i, (w, b) in enumerate(zip(self.dec_weights, self.dec_biases)):
            h = h @ w + b
            if i < len(self.dec_weights) - 1:
                h = self._relu(h)

        return h

    def forward(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from prior."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)

    def interpolate(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        n_steps: int = 10,
    ) -> List[np.ndarray]:
        """Interpolate between two prompts in latent space."""
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)

        interpolations = []
        for t in np.linspace(0, 1, n_steps):
            z = (1 - t) * mu1 + t * mu2
            x_interp = self.decode(z)
            interpolations.append(x_interp)

        return interpolations


# =============================================================================
# Denoising Diffusion for Prompt Generation
# =============================================================================


class PromptDiffusion:
    """Denoising Diffusion Model for prompt generation.

    Generates prompts by iteratively denoising from random noise.
    """

    def __init__(
        self,
        dim: int = 256,
        n_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.dim = dim
        self.n_steps = n_steps

        # Noise schedule
        self.betas = np.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        # Simple denoiser (MLP)
        hidden = 512
        scale = np.sqrt(2.0 / dim)
        self.w1 = np.random.randn(dim + 1, hidden) * scale  # +1 for timestep
        self.b1 = np.zeros(hidden)
        self.w2 = np.random.randn(hidden, hidden) * scale
        self.b2 = np.zeros(hidden)
        self.w3 = np.random.randn(hidden, dim) * scale
        self.b3 = np.zeros(dim)

    def _denoise(self, x: np.ndarray, t: int) -> np.ndarray:
        """Predict noise at timestep t."""
        # Concatenate timestep
        t_embed = np.array([t / self.n_steps]).reshape(1, 1)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_t = np.concatenate([x, np.tile(t_embed, (x.shape[0], 1))], axis=-1)

        # Forward through network
        h = np.maximum(0, x_t @ self.w1 + self.b1)
        h = np.maximum(0, h @ self.w2 + self.b2)
        noise_pred = h @ self.w3 + self.b3

        return noise_pred

    def add_noise(
        self,
        x: np.ndarray,
        t: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise at timestep t."""
        alpha_bar = self.alpha_bars[t]
        noise = np.random.randn(*x.shape)
        x_noisy = np.sqrt(alpha_bar) * x + np.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise

    def sample(
        self,
        n_samples: int = 1,
        guidance_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        guidance_scale: float = 1.0,
    ) -> np.ndarray:
        """Generate samples via iterative denoising."""
        # Start from noise
        x = np.random.randn(n_samples, self.dim)

        for t in reversed(range(self.n_steps)):
            # Predict noise
            noise_pred = self._denoise(x, t)

            # Apply guidance if provided
            if guidance_fn is not None:
                grad = guidance_fn(x)
                noise_pred = noise_pred - guidance_scale * grad

            # Denoise step
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            if t > 0:
                alpha_bar_prev = self.alpha_bars[t - 1]
                noise = np.random.randn(*x.shape)
            else:
                alpha_bar_prev = 1.0
                noise = 0

            # DDPM update
            x = (1 / np.sqrt(alpha)) * (
                x - (beta / np.sqrt(1 - alpha_bar)) * noise_pred
            ) + np.sqrt(beta) * noise

        return x


# =============================================================================
# Reinforcement Learning for Prompt Optimization
# =============================================================================


@dataclass
class RLExperience:
    """Single RL experience."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class PromptPolicyGradient:
    """Policy Gradient for prompt optimization.

    Learns a policy that outputs prompt modifications.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma

        # Policy network
        scale = np.sqrt(2.0 / state_dim)
        self.w1 = np.random.randn(state_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.w2_mu = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b2_mu = np.zeros(action_dim)
        self.w2_logstd = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b2_logstd = np.zeros(action_dim) - 1  # Start with smaller variance

        # Experience buffer
        self.experiences: List[RLExperience] = []

    def _forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass to get action distribution."""
        h = np.tanh(state @ self.w1 + self.b1)
        mu = h @ self.w2_mu + self.b2_mu
        log_std = h @ self.w2_logstd + self.b2_logstd
        log_std = np.clip(log_std, -20, 2)
        return mu, log_std

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action from policy."""
        mu, log_std = self._forward(state)

        if deterministic:
            return mu

        std = np.exp(log_std)
        action = mu + std * np.random.randn(*mu.shape)
        return action

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience."""
        self.experiences.append(RLExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        ))

    def update(self) -> float:
        """Update policy using REINFORCE."""
        if not self.experiences:
            return 0.0

        # Compute returns
        returns = []
        G = 0
        for exp in reversed(self.experiences):
            G = exp.reward + self.gamma * G * (1 - exp.done)
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute gradients
        total_loss = 0
        for exp, ret in zip(self.experiences, returns):
            mu, log_std = self._forward(exp.state.reshape(1, -1))
            std = np.exp(log_std)

            # Log probability
            log_prob = -0.5 * ((exp.action - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
            log_prob = log_prob.sum()

            # Policy gradient
            loss = -log_prob * ret
            total_loss += loss

            # Simplified gradient update
            h = np.tanh(exp.state @ self.w1 + self.b1)
            # Gradient of loss w.r.t. mu: (action_dim,)
            grad_output = (-ret * (exp.action - mu.flatten()) / (std.flatten() ** 2))
            # Gradient w.r.t. w2_mu: outer(h, grad_output)
            self.w2_mu -= self.lr * np.outer(h, grad_output)
            self.b2_mu -= self.lr * grad_output

        self.experiences.clear()
        return float(total_loss)


class PromptPPO:
    """Proximal Policy Optimization for stable prompt learning."""

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 0.0003,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        n_epochs: int = 10,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.n_epochs = n_epochs

        # Actor
        scale = np.sqrt(2.0 / state_dim)
        self.actor_w1 = np.random.randn(state_dim, hidden_dim) * scale
        self.actor_b1 = np.zeros(hidden_dim)
        self.actor_w2 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.actor_b2 = np.zeros(action_dim)
        self.log_std = np.zeros(action_dim)

        # Critic
        self.critic_w1 = np.random.randn(state_dim, hidden_dim) * scale
        self.critic_b1 = np.zeros(hidden_dim)
        self.critic_w2 = np.random.randn(hidden_dim, 1) * scale
        self.critic_b2 = np.zeros(1)

        # Buffer
        self.buffer: List[Dict] = []

    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get action, log_prob, and value."""
        # Actor
        h = np.tanh(state @ self.actor_w1 + self.actor_b1)
        mu = h @ self.actor_w2 + self.actor_b2
        std = np.exp(self.log_std)

        action = mu + std * np.random.randn(*mu.shape)
        log_prob = -0.5 * (((action - mu) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum()

        # Critic
        h_v = np.tanh(state @ self.critic_w1 + self.critic_b1)
        value = float((h_v @ self.critic_w2 + self.critic_b2)[0])

        return action, np.array([log_prob]), value

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: np.ndarray,
    ) -> None:
        """Store transition."""
        self.buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "value": value,
            "log_prob": log_prob,
        })

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Update policy with PPO."""
        if len(self.buffer) < 2:
            return {}

        # Compute advantages
        rewards = [t["reward"] for t in self.buffer]
        values = [t["value"] for t in self.buffer] + [last_value]

        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + np.array([t["value"] for t in self.buffer])

        # PPO update
        total_loss = 0
        for _ in range(self.n_epochs):
            for i, transition in enumerate(self.buffer):
                state = transition["state"]
                action = transition["action"]
                old_log_prob = transition["log_prob"]
                adv = advantages[i]
                ret = returns[i]

                # Current policy
                h = np.tanh(state @ self.actor_w1 + self.actor_b1)
                mu = h @ self.actor_w2 + self.actor_b2
                std = np.exp(self.log_std)

                log_prob = -0.5 * (((action - mu) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi))
                log_prob = log_prob.sum()

                # Ratio
                ratio = np.exp(log_prob - old_log_prob)

                # Clipped objective
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                actor_loss = -min(surr1, surr2)

                # Critic loss
                h_v = np.tanh(state @ self.critic_w1 + self.critic_b1)
                value = float((h_v @ self.critic_w2 + self.critic_b2)[0])
                critic_loss = 0.5 * (ret - value) ** 2

                total_loss += actor_loss + 0.5 * critic_loss

        self.buffer.clear()
        return {"loss": float(total_loss)}


# =============================================================================
# Meta-Learning for Few-Shot Adaptation
# =============================================================================


class MAML:
    """Model-Agnostic Meta-Learning for prompt adaptation.

    Learns initialization that can quickly adapt to new tasks.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 64,
        hidden_dim: int = 128,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps

        # Initialize meta-parameters
        scale = np.sqrt(2.0 / input_dim)
        self.w1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) * scale
        self.b2 = np.zeros(output_dim)

    def _forward(
        self,
        x: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> np.ndarray:
        """Forward pass with given parameters."""
        h = np.maximum(0, x @ w1 + b1)
        return h @ w2 + b2

    def _inner_update(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Inner loop update on support set."""
        for _ in range(self.n_inner_steps):
            # Forward
            h = np.maximum(0, support_x @ w1 + b1)
            pred = h @ w2 + b2

            # Loss gradient
            grad_output = 2 * (pred - support_y) / len(support_x)

            # Backprop
            grad_w2 = h.T @ grad_output
            grad_b2 = grad_output.sum(axis=0)

            grad_h = grad_output @ w2.T
            grad_h = grad_h * (h > 0)  # ReLU gradient

            grad_w1 = support_x.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)

            # Update
            w1 = w1 - self.inner_lr * grad_w1
            b1 = b1 - self.inner_lr * grad_b1
            w2 = w2 - self.inner_lr * grad_w2
            b2 = b2 - self.inner_lr * grad_b2

        return w1, b1, w2, b2

    def meta_train(
        self,
        tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> float:
        """Meta-training step on batch of tasks.

        Each task is (support_x, support_y, query_x, query_y).
        """
        meta_grad_w1 = np.zeros_like(self.w1)
        meta_grad_b1 = np.zeros_like(self.b1)
        meta_grad_w2 = np.zeros_like(self.w2)
        meta_grad_b2 = np.zeros_like(self.b2)

        total_loss = 0

        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop
            w1, b1, w2, b2 = self._inner_update(
                support_x, support_y,
                self.w1.copy(), self.b1.copy(),
                self.w2.copy(), self.b2.copy(),
            )

            # Query loss
            pred = self._forward(query_x, w1, b1, w2, b2)
            loss = np.mean((pred - query_y) ** 2)
            total_loss += loss

            # Accumulate meta-gradients (simplified)
            grad_output = 2 * (pred - query_y) / len(query_x)
            h = np.maximum(0, query_x @ w1 + b1)

            meta_grad_w2 += h.T @ grad_output
            meta_grad_b2 += grad_output.sum(axis=0)

            grad_h = grad_output @ w2.T * (h > 0)
            meta_grad_w1 += query_x.T @ grad_h
            meta_grad_b1 += grad_h.sum(axis=0)

        # Meta update
        self.w1 -= self.outer_lr * meta_grad_w1 / len(tasks)
        self.b1 -= self.outer_lr * meta_grad_b1 / len(tasks)
        self.w2 -= self.outer_lr * meta_grad_w2 / len(tasks)
        self.b2 -= self.outer_lr * meta_grad_b2 / len(tasks)

        return float(total_loss / len(tasks))

    def adapt(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Adapt to new task and return prediction function."""
        w1, b1, w2, b2 = self._inner_update(
            support_x, support_y,
            self.w1.copy(), self.b1.copy(),
            self.w2.copy(), self.b2.copy(),
        )

        def predict(x: np.ndarray) -> np.ndarray:
            return self._forward(x, w1, b1, w2, b2)

        return predict


# =============================================================================
# Unified APEX Transformer
# =============================================================================


class APEXTransformer:
    """Unified APEX Transformer for prompt manipulation.

    Combines all neural architectures:
    - Multi-head attention for sequence modeling
    - Graph attention for structure
    - VAE for latent space
    - Diffusion for generation
    - RL for optimization
    - Meta-learning for adaptation
    """

    def __init__(
        self,
        embed_dim: int = 256,
        latent_dim: int = 64,
        n_heads: int = 8,
    ):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Core components
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.graph_attn = PromptGraphAttention(embed_dim, embed_dim // 2)
        self.vae = PromptVAE(embed_dim, latent_dim)
        self.diffusion = PromptDiffusion(embed_dim)
        self.policy = PromptPPO(embed_dim, latent_dim)
        self.meta = MAML(embed_dim, latent_dim)

    def encode(self, prompt_embedding: np.ndarray) -> np.ndarray:
        """Encode prompt to latent space."""
        mu, _ = self.vae.encode(prompt_embedding.reshape(1, -1))
        return mu.squeeze()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent to prompt embedding."""
        return self.vae.decode(latent.reshape(1, -1)).squeeze()

    def attend(self, sequence: np.ndarray) -> np.ndarray:
        """Apply attention to prompt sequence."""
        output, _ = self.attention.forward(sequence)
        return output

    def generate(
        self,
        n_samples: int = 1,
        guidance_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """Generate new prompt embeddings."""
        return self.diffusion.sample(n_samples, guidance_fn)

    def interpolate(
        self,
        embed1: np.ndarray,
        embed2: np.ndarray,
        n_steps: int = 10,
    ) -> List[np.ndarray]:
        """Interpolate between two prompts."""
        return self.vae.interpolate(
            embed1.reshape(1, -1),
            embed2.reshape(1, -1),
            n_steps,
        )

    def optimize_step(
        self,
        state: np.ndarray,
        reward: float,
    ) -> np.ndarray:
        """Take RL optimization step."""
        action, log_prob, value = self.policy.get_action(state)
        self.policy.store(state, action, reward, value, log_prob)
        return state + action  # New state

    def adapt_to_task(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Callable:
        """Adapt to new task with few examples."""
        support_x = np.array([e[0] for e in examples])
        support_y = np.array([e[1] for e in examples])
        return self.meta.adapt(support_x, support_y)

    def get_statistics(self) -> Dict[str, Any]:
        """Get component statistics."""
        return {
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "vae_input_dim": self.vae.input_dim,
            "diffusion_steps": self.diffusion.n_steps,
            "policy_buffer_size": len(self.policy.buffer),
        }
