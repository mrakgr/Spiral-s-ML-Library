let arg_max (l : int list) =
    l
    |> List.mapi (fun i x -> x, System.Random.Shared.NextDouble(), i)
    |> List.max
    |> fun (_,_,i) -> i

arg_max [1;2;3;4;5;5;5;5;5;5]

