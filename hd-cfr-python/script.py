from typing import Union, Literal, Tuple

card = Literal[
    "King",
    "Queen",
    "Jack",
]
action = Literal[
    "Raise",
    "Fold",
    "Call",
]
game_trace = Union[
    Tuple[Literal["Act"], action],
    Tuple[Literal["Draw"], card],
]

k : list[game_trace] = [ ("Draw", "King"), ("Act", "Raise"), ("Act", "Call"), ("Draw", "Queen") ]
v : list[float] =  [2., 1., -1.]

d = {k: v}
