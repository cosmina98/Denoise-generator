#!/usr/bin/env python
"""Provides scikit interface."""
import numpy as np


def _remove_duplicates(costs, items):
    dedup_costs = []
    dedup_items = []
    costs = [tuple(c) for c in costs]
    prev_c = None
    for c, g in sorted(zip(costs, items), key=lambda x: x[0]):
        if prev_c != c:
            dedup_costs.append(c)
            dedup_items.append(g)
            prev_c = c
    return np.array(dedup_costs), dedup_items


def _is_pareto_efficient(costs):
    is_eff = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_eff[i]:
            is_eff[i] = False
            # Remove dominated points
            is_eff[is_eff] = np.any(costs[is_eff] < c, axis=1)
            is_eff[i] = True
    return is_eff


def _pareto_front(costs):
    return [i for i, p in enumerate(_is_pareto_efficient(costs)) if p]


def _pareto_set(items, costs, return_costs=False, return_complementary=False):
    return_item = []
    ids = _pareto_front(costs)
    select_items = [items[i] for i in ids]
    return_item.append(select_items)
    if return_costs:
        select_costs = np.array([costs[i] for i in ids])
        return_item.append(select_costs)
    if return_complementary:
        comp_ids = [i for i in range(len(costs)) if i not in ids]
        comp_costs = np.array([costs[i] for i in comp_ids])
        comp_items = [items[i] for i in comp_ids]
        return_item.append(comp_items)
        if return_costs:
            return_item.append(comp_costs)
    if len(return_item) == 1:
        return return_item[0]
    return tuple(return_item)


def get_pareto_set(items, costs, return_costs=False, return_complementary=False):
    """get_pareto_set."""
    costs, items = _remove_duplicates(costs, items)
    return _pareto_set(items, costs, return_costs, return_complementary)


def pareto_shells(items, costs, n_levels=None):
    """pareto_shells."""
    sel_items_list = []
    sel_costs_list = []
    if n_levels is None:
        n_levels = len(items)
    curr_items = items
    curr_costs = costs
    for i in range(n_levels):
        if curr_items:
            sel_items, sel_costs, comp_items, comp_costs = get_pareto_set(curr_items, curr_costs, return_costs=True, return_complementary=True)
            sel_items_list.append(sel_items)
            sel_costs_list.append(sel_costs)
            curr_items = comp_items
            curr_costs = comp_costs
    return sel_items_list, sel_costs_list


def pareto_select(items, costs, n_items=None):
    """pareto_shells."""
    if n_items is None:
        n_items = len(items)
    sel_items_list = []
    sel_costs_list = []
    curr_items = items
    curr_costs = costs
    curr_n_items = 0
    for i in range(len(items)):
        if curr_items:
            sel_items, sel_costs, comp_items, comp_costs = get_pareto_set(curr_items, curr_costs, return_costs=True, return_complementary=True)
            curr_n_items += len(sel_items)
            sel_items_list.append(sel_items)
            sel_costs_list.append(sel_costs)
            curr_items = comp_items
            curr_costs = comp_costs
        if curr_n_items >= n_items or curr_n_items == len(items):
            break
    return sel_items_list, sel_costs_list
