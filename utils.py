import os
from functools import partial
from typing import Dict

import jax
import jax.random as rax
import numpy as np
from jax import numpy as jp
from typing import Optional, Union, Tuple, Any

Array = Any


def slurm_infos():
    return {
        'slurm/job_id': os.getenv('SLURM_JOB_ID'),
        'slurm/job_user': os.getenv('SLURM_JOB_USER'),
        'slurm/job_partition': os.getenv('SLURM_JOB_PARTITION'),
        'slurm/cpus_per_node': os.getenv('SLURM_JOB_CPUS_PER_NODE'),
        'slurm/num_nodes': os.getenv('SLURM_JOB_NUM_NODES'),
        'slurm/nodelist': os.getenv('SLURM_JOB_NODELIST'),
        'slurm/cluster_name': os.getenv('SLURM_CLUSTER_NAME'),
        'slurm/array_task_id': os.getenv('SLURM_ARRAY_TASK_ID')
    }


def rscope(rng, *path):
    if rng.ndim > 1:  # deal with leading batch axes
        return jax.vmap(lambda rng: rscope(rng, *path))(rng)
    #  NOTE used to use seed = hash(path) but this is nondeterministic
    import zlib
    data = "/".join(path).encode("ascii")
    seed = zlib.crc32(data)
    return rax.fold_in(rng, seed)


def clip_grads(grads, max_grad):
    return jax.tree_map(lambda dx: jp.clip(dx, -max_grad, +max_grad), grads)


def clip_grads_by_norm(updates, max_norm):
    # taken from https://github.com/deepmind/optax/blob/9dbf9366996c4daeaf0bdc8394aa3f79a7946949/optax/_src/clipping.py
    g_norm = global_norm(updates)
    trigger = jp.squeeze(g_norm < max_norm)

    def clip_fn(t):
        return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_norm)

    updates = jax.tree_util.tree_map(clip_fn, updates)
    return updates


def global_norm(updates):
    # taken from https://github.com/deepmind/optax/blob/9dbf9366996c4daeaf0bdc8394aa3f79a7946949/optax/_src/clipping.py
    return jp.sqrt(sum(jp.sum(x ** 2) for x in jax.tree_util.tree_leaves(updates)))


def npify(tree):
    return jax.tree_map(lambda p: np.array(p), tree)


class AliasDict(dict):
    def __init__(self, redirects: Dict[str, str], *args, **kwargs):
        self.redirects = redirects
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key in self.redirects:
            key = self.redirects[key]
        return super().__getitem__(key)


@partial(jax.jit, static_argnames=("axis",))
def log_softmax_with_stop_grad_normalizing_constant(x: Array,
                                                    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
                                                    where: Optional[Array] = None,
                                                    initial: Optional[Array] = None) -> Array:
    """This is taken from /jax/_src/nn/functions.py but changed to stop grad at the normalizing constant"""

    x_max = jp.max(x, axis, where=where, initial=initial, keepdims=True)
    shifted = x - jax.lax.stop_gradient(x_max)
    shifted_logsumexp = jp.log(
        jp.sum(jp.exp(shifted), axis, where=where, keepdims=True))
    result = shifted - jax.lax.stop_gradient(shifted_logsumexp)
    if where is not None:
        return jp.where(where, result, -jp.inf)
    return result
