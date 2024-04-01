import time
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


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
        y = nn.Conv(
            features=self.channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False
        )(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = mish(y)
        y = nn.Conv(
            features=self.channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False
        )(x)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if self.se:
            squeeze = jnp.mean(y, axis=(1, 2), keepdims=True)

            excitation = nn.Dense(
                features=self.channels // self.se_ratio, use_bias=True
            )(squeeze)
            excitation = nn.relu(excitation)
            excitation = nn.Dense(features=self.channels, use_bias=True)(excitation)
            excitation = nn.hard_sigmoid(excitation)

            y = y * excitation

        return mish(x + y)


class AZResnet(nn.Module):
    config: AZResnetConfig

    @nn.compact
    def __call__(self, x, train: bool):
        batch_size = x.shape[0]

        x = nn.Conv(
            features=self.config.channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = mish(x)

        for _ in range(self.config.num_blocks):
            x = ResidualBlock(channels=self.config.channels, se=True)(x, train=train)

        # policy head
        a = nn.Conv(
            features=self.config.channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            use_bias=False,
        )(x)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = mish(a)
        a = nn.Conv(
            features=self.config.policy_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            use_bias=False,
        )(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = mish(a)
        a = a.reshape((batch_size, -1))
        a = nn.Dense(features=self.config.num_policy_labels)(a)

        b = nn.Conv(
            features=self.config.channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            use_bias=False,
        )(x)
        b = nn.BatchNorm(use_running_average=not train)(b)
        b = mish(b)
        b = nn.Conv(
            features=self.config.policy_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            use_bias=False,
        )(b)
        b = nn.BatchNorm(use_running_average=not train)(b)
        b = mish(b)
        b = b.reshape((batch_size, -1))
        b = nn.Dense(features=self.config.num_policy_labels)(b)

        # Sit Action
        c = nn.Conv(
            features=self.config.channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            use_bias=False,
        )(x)
        c = nn.BatchNorm(use_running_average=not train)(c)
        c = mish(c)
        c = nn.Conv(
            features=self.config.policy_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            use_bias=False,
        )(c)
        c = nn.BatchNorm(use_running_average=not train)(c)
        c = mish(c)
        c = a.reshape((batch_size, -1))
        c = nn.Dense(features=1)(c)
        policy = jnp.concatenate((a, b, c), axis=1)

        # value head
        value = nn.Conv(
            features=self.config.value_channels, kernel_size=(1, 1), use_bias=False
        )(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = mish(value)
        value = value.reshape((batch_size, -1))
        value = nn.Dense(features=256)(value)
        value = mish(value)
        value = nn.Dense(features=1)(value)
        value = nn.tanh(value)
        value = value.squeeze(axis=1)

        return policy, value


if __name__ == "__main__":
    from itertools import product
    from functools import partial

    model = AZResnet(
        AZResnetConfig(
            num_blocks=15,
            channels=256,
            policy_channels=4,
            value_channels=8,
            num_policy_labels=64*77,
        )
    )
    x = jnp.ones((1, 8, 16, 32))
    variables = model.init(jax.random.key(0), x, train=False)

    forward = jax.jit(partial(model.apply, variables, train=False))

    for i in range(100):
        start = time.time()
        out = forward(x)
        print(time.time() - start)
    policy, value = out
    print(policy.shape)