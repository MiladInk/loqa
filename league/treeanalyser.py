# a custom library I wrote for analysing the gradients of a flax model w.r.t to specific subparts of the model
# it's not very well documented, but it's pretty simple to use

import flax
import jax, jax.numpy as jp
from flax.struct import PyTreeNode
from typing import Dict


def analyse_tree(tree: PyTreeNode,
                 named_masks: Dict[str, PyTreeNode],
                 key_prefix: str,
                 melt_output: bool = True,
                 ) -> Dict[str, Dict[str, PyTreeNode]]:
    stats = {}
    for name, mask in named_masks.items():
        tree_extract = extract_just_mask(tree, mask)
        assert tree_extract.shape[0] > 0, 'mask is empty'
        stats[name] = {
            "mean": jp.mean(tree_extract),
            "std": jp.std(tree_extract),
            "min": jp.min(tree_extract),
            "max": jp.max(tree_extract),
            "max_abs": jp.max(jp.abs(tree_extract)),
            "norm": jp.linalg.norm(tree_extract),
        }
    if melt_output:
        stats = melted_dict(stats)
    if key_prefix:
        stats = prepend_keys(stats, key_prefix)
    return stats

def named_maskfuncs_to_named_masks(tree, named_maskfuncs):
    named_masks = {}
    for name, maskfunc in named_maskfuncs.items():
        named_masks[name] = get_mask(tree, maskfunc)
    return named_masks

def get_mask(tree, mask_fun):
    mask = flattened_traversal(mask_fun)(tree)
    mask = flax.core.frozen_dict.FrozenDict(mask)
    return mask

def get_mask_for_key_func(key: str):
    def one_for_key_zero_for_other(path, value):
        return jp.where(key in path, jp.ones_like(value), jp.zeros_like(value))

    return one_for_key_zero_for_other

def get_all_ones_mask_func():
    def all_ones(path, value):
        return jp.ones_like(value)

    return all_ones

def flattened_traversal(fn):
  """Returns function that is called with `(path, param)` instead of pytree."""

  def mask(tree):
      flat = flax.traverse_util.flatten_dict(tree)
      return flax.traverse_util.unflatten_dict(
          {k: fn(k, v) for k, v in flat.items()})

  return mask

# unit test for all ones mask
def test_all_ones_mask():
    tree = {'a': 1, 'b': 2}
    mask = get_mask(tree, get_all_ones_mask_func())
    assert mask == {'a': 1, 'b': 1}

def test_mask_for_key():
    tree = {'a': 1, 'b': 2}
    mask = get_mask(tree, get_mask_for_key_func('a'))
    assert mask == {'a': 1, 'b': 0}

    tree = {'a': 1, 'b': {'c': 2, 'd': 3}}
    mask = get_mask(tree, get_mask_for_key_func('c'))
    assert mask == {'a': 0, 'b': {'c': 1, 'd': 0}}
    mask = get_mask(tree, get_mask_for_key_func('b'))
    assert mask == {'a': 0, 'b': {'c': 1, 'd': 1}}

def extract_just_mask(tree, mask):
    flat_mask = flatten_tree(mask)
    flat_tree = flatten_tree(tree)
    return flat_tree[flat_mask == 1]

def flatten_tree(tree):
    return jp.concatenate(jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.reshape(-1), tree)))

def melted_dict(d):
    return {'_'.join(k): v for k, v in flax.traverse_util.flatten_dict(d).items()}

def prepend_keys(d, prefix):
    return {prefix + '_' + k: v for k, v in d.items()}

def test_extract():
    tree = {'a': jp.array(1), 'b': jp.array(17)}
    mask = get_mask(tree, get_all_ones_mask_func())
    assert jp.allclose(extract_just_mask(tree, mask), jp.array([1, 17]))
    mask = get_mask(tree, get_mask_for_key_func('a'))
    assert extract_just_mask(tree, mask) == jp.array([1])
    mask = get_mask(tree, get_mask_for_key_func('b'))
    assert extract_just_mask(tree, mask) == jp.array([17])

    tree = {'a': jp.array(1), 'b': {'c': jp.array(2), 'd': jp.array(3)}}
    mask = get_mask(tree, get_all_ones_mask_func())
    assert jp.allclose(extract_just_mask(tree, mask), jp.array([1, 2, 3]))
    mask = get_mask(tree, get_mask_for_key_func('a'))
    assert extract_just_mask(tree, mask) == jp.array([1])
    mask = get_mask(tree, get_mask_for_key_func('b'))
    assert jp.allclose(extract_just_mask(tree, mask), jp.array([2, 3]))
    mask = get_mask(tree, get_mask_for_key_func('c'))
    assert extract_just_mask(tree, mask) == jp.array([2])
    mask = get_mask(tree, get_mask_for_key_func('d'))
    assert extract_just_mask(tree, mask) == jp.array([3])

def test_analyze_tree():
    tree = {'a': jp.array(1), 'b': {'c': jp.array(2), 'd': jp.array(3)}}
    mask1 = get_mask(tree, get_all_ones_mask_func())
    mask2 = get_mask(tree, get_mask_for_key_func('a'))
    mask3 = get_mask(tree, get_mask_for_key_func('b'))
    mask4 = get_mask(tree, get_mask_for_key_func('c'))
    mask5 = get_mask(tree, get_mask_for_key_func('d'))
    named_masks = {'all': mask1, 'a': mask2, 'b': mask3, 'c': mask4, 'd': mask5}
    stats = analyse_tree(tree, named_masks)
    assert stats['all']['mean'] == 2
    assert stats['all']['std'] == jp.sqrt(jp.array(2)/3)
    assert stats['all']['min'] == 1
    assert stats['all']['max'] == 3
    assert stats['all']['max_abs'] == 3
    assert stats['a']['mean'] == 1
    assert stats['a']['std'] == 0
    assert stats['a']['min'] == 1
    assert stats['a']['max'] == 1
    assert stats['a']['max_abs'] == 1
    assert stats['b']['mean'] == 2.5
    assert stats['b']['std'] == jp.sqrt(jp.array(0.5**2+0.5**2)/2)
    assert stats['b']['min'] == 2
    assert stats['b']['max'] == 3
    assert stats['b']['max_abs'] == 3
    assert stats['c']['mean'] == 2
    assert stats['c']['std'] == 0
    assert stats['c']['min'] == 2
    assert stats['c']['max'] == 2
    assert stats['c']['max_abs'] == 2
    assert stats['d']['mean'] == 3
    assert stats['d']['std'] == 0
    assert stats['d']['min'] == 3
    assert stats['d']['max'] == 3
    assert stats['d']['max_abs'] == 3



if __name__ == '__main__':
    test_all_ones_mask()
    test_mask_for_key()
    test_extract()
    test_analyze_tree()

