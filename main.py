# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
Wildfire propagation simulation
"""

import argparse

import torch
from PIL import Image
from torch.nn.functional import max_pool2d

# States (0: clear, 1: fuel, 2: fire, 3: burnt)
ALLOWED_STATES = [0, 1, 2, 3]


def state_transition(prev_state: torch.Tensor, ignition_prob: float = 1, combustion_prob: float = 1) -> torch.Tensor:
    masks = torch.rand((2, *prev_state.shape))
    # Light up nearby fuel
    is_fuel = prev_state == 1
    near_fire = max_pool2d(
        (prev_state == 2).float().unsqueeze(0).unsqueeze(0),
        (3, 3),
        padding=1,
        stride=1,
    ).to(dtype=torch.bool).squeeze(0).squeeze(0)
    state = prev_state.clone()
    state[is_fuel & near_fire & (masks[0] <= ignition_prob)] = 2
    # Put out old burning cells
    state[(masks[1] <= combustion_prob) & (prev_state == 2)] = 3

    return state


def main(args):

    # cf. article
    # https://iopscience.iop.org/article/10.1088/1742-6596/285/1/012038/pdf

    size = (args.map_size, args.map_size)
    states = torch.zeros((args.it, *size), dtype=torch.uint8)
    states[0] = torch.rand(size) <= args.fuel_prob
    # Set a point on fire
    states[0, size[0] // 2, size[1] // 2] = 2

    # Simulate all steps
    for t in range(1, args.it):
        states[t] = state_transition(states[t - 1], args.ignition_prob, args.extinction_prob)

    # Viz: color code states
    colored = torch.zeros((args.it, *size, 3), dtype=torch.uint8)
    # Clear
    colored[states == 0] = torch.tensor((139, 69, 19), dtype=torch.uint8)
    # Fuel
    colored[states == 1] = torch.tensor((0, 255, 0), dtype=torch.uint8)
    # Burning
    colored[states == 2] = torch.tensor((255, 0, 0), dtype=torch.uint8)

    imgs = [Image.fromarray(img) for img in colored.numpy()]
    imgs[0].save('tmp.gif', format="GIF", append_images=imgs[1:], save_all=True, duration=100, loop=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wildfire propagation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("it", type=int, help="Number of time steps")
    parser.add_argument("--output-file", type=str, default="./simulation.gif", help="Where the GIF will be saved")
    # Environment generation
    parser.add_argument("--map-size", type=int, default=512, help="Number of cells on each side of the map")
    parser.add_argument("--fuel-prob", type=float, default=.6, help="Probability that a cell has fuel")
    # Combustion parameters
    parser.add_argument(
        "--ignition-prob",
        type=float,
        default=.95,
        help="Probability that a burning cell will ignite a nearbouring cell",
    )
    parser.add_argument(
        "--extinction-prob",
        type=float,
        default=.8,
        help="Probability that the combustion will terminate within one iteration",
    )
    args = parser.parse_args()

    main(args)
