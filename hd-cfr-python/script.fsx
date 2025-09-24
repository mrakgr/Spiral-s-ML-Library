open System
open System.Collections.Generic

type t =
    | String of string
    | StringSet of string Set

type card =
    | King
    | Queen
    | Jack

type action =
    | Raise
    | Call
    | Fold

type game_trace =
    | Act of action
    | Draw of card

let d = 
    [
        [ Draw King; Act Raise; Act Call; Draw Queen ], [2.; 1.; -1.]
    ]
    |> dict