from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn


def mish(x): return x * jnp.tanh(jax.nn.softplus(x))

@dataclass
class AZResnetConfig:
    num_blocks: int
    channels: int
    policy_channels: int
    value_channels: int
    num_policy_labels: int

class ResidualBlock(nn.Module):
    channels: int
    se: bool
    se_ratio: int = 4

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(features=self.channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = mish(y)
        y = nn.Conv(features=self.channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if self.se: 
            squeeze = jnp.mean(y, axis=(1, 2), keepdims=True)

            excitation = nn.Dense(features=self.channels // self.se_ratio, use_bias=True)(squeeze)
            excitation = nn.relu(excitation)
            excitation = nn.Dense(features=self.channels, use_bias=True)(excitation)
            excitation = nn.hard_sigmoid(excitation)

            y = y * excitation

        return mish(x + y)

class AZResnet(nn.Module):
    config: AZResnetConfig

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=self.config.channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = mish(x)

        for _ in range(self.config.num_blocks):
            x = ResidualBlock(channels=self.config.channels, se=True)(x, train=train)

        # policy head
        policy = [None, None]
        policy[0] = nn.Conv(features=self.config.channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False)(x)
        policy[0] = nn.BatchNorm(use_running_average=not train)(policy[0])
        policy[0] = mish(policy[0])
        policy[0] = nn.Conv(features=self.config.policy_channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False)(policy[0])
        policy[0] = nn.BatchNorm(use_running_average=not train)(policy[0])
        policy[0] = mish(policy[0])
        policy[0] = policy[0].reshape((policy[0].shape[0], -1))
        policy[0] = nn.Dense(features=self.config.num_policy_labels)(policy[0])

        policy[1] = nn.Conv(features=self.config.channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False)(x)
        policy[1] = nn.BatchNorm(use_running_average=not train)(policy[1])
        policy[1] = mish(policy[1])
        policy[1] = nn.Conv(features=self.config.policy_channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False)(policy[1])
        policy[1] = nn.BatchNorm(use_running_average=not train)(policy[1])
        policy[1] = mish(policy[1])
        policy[1] = policy[1].reshape((policy[1].shape[0], -1))
        policy[1] = nn.Dense(features=self.config.num_policy_labels)(policy[1])

        # value head
        value = nn.Conv(features=self.config.value_channels, kernel_size=(1, 1), use_bias=False)(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = mish(value)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(features=256)(value)
        value = mish(value)
        value = nn.Dense(features=1)(value)
        value = nn.tanh(value)

        return policy, value

