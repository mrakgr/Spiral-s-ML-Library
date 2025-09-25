open System

let intToBinary (value: int) : string =
    Convert.ToString(value, 2)

let deck = (1 <<< 6) - 1

intToBinary deck