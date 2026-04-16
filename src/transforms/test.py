import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from transforms import LandMarksRandHorizontalFlipNeighbor, LandMarksRandHorizontalFlip

poi_flip_pairs = {
    # These are the middle points, i.e. the ones that are not flipped
    81: 81,
    101: 101,
    103: 103,
    102: 102,
    104: 104,
    105: 105,  #
    106: 106,  #
    107: 107,  #
    108: 108,  #
    125: 125,
    127: 127,
    # Flipped left to right
    83: 82,
    84: 85,
    86: 87,
    88: 89,
    109: 117,
    111: 119,
    110: 118,
    112: 120,
    113: 121,  #
    114: 122,  #
    115: 123,  #
    116: 124,  #
    # Flipped right to left
    82: 83,
    85: 84,
    87: 86,
    89: 88,
    117: 109,
    118: 110,
    119: 111,
    120: 112,
    121: 113,
    122: 114,
    123: 115,
    124: 116,
    # Center of mass, does not need to be flipped
    # TODO: Passt das so??? geht das auch wenn include_com=false ist und die POIs gar nicht definiert sind?
    41: 41,
    42: 42,
    43: 43,
    44: 44,
    45: 45,
    46: 46,
    47: 47,
    48: 48,
    49: 49,
    50: 50,
    0: 0,
}

if __name__ == "__main__":
    import torch

    flipper = LandMarksRandHorizontalFlipNeighbor(
        prob=1.0,
        flip_pairs=poi_flip_pairs,
    )

    input = torch.zeros((1, 10, 10, 10))
    target_indices = torch.tensor(
        [
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            127,
        ]
    )

    target_indices = torch.cat([target_indices, target_indices, target_indices])
    target = torch.zeros((len(target_indices), 3))
    target[:, 0] = torch.arange(0, len(target_indices))

    print("input target:", target.shape)

    dd = {"input": input, "target": target, "target_indices": target_indices}

    out = flipper(dd)
    print("output target:", out["target"][:10])

    # [0, 2, 1, 4, 3, 6, 5, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32, 17, 18, 19, 20, 21, 22, 23, 24, 33, 34]
