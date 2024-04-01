import jax
import jax.numpy as jnp 
from training.constants import POLICY_LABELS, BOARD_HEIGHT, BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS

from training.architectures.azresnet import AZResnet, AZResnetConfig
from training.trainer import TrainerModule

x=jnp.ones((1, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS))
trainer = TrainerModule(model_name='AZResNet', model_class=AZResnet, model_configs=AZResnetConfig(
    num_blocks=15,
    channels=256,
    policy_channels=4, 
    value_channels=8,
    num_policy_labels=len(POLICY_LABELS)
), optimizer_name='lion', optimizer_params={'learning_rate': 0.00001}, x=x)
trainer.init_optimizer()
trainer.save_checkpoint(trainer.state, step=1)

state = trainer.load_checkpoint('5')
variables = {'params': state['params'], 'batch_stats': state['batch_stats']}

model = AZResnet(
    AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4,
        value_channels=8,
        num_policy_labels=2185,
    )
)
#init_rng = jax.random.PRNGKey(0)
#variables = model.init(init_rng, x, train=True)
print(variables.keys())
policy_logits, value_logits = model.apply(variables, x, train=False)
print(value_logits)