let x = 
    [1;2;3;4;5] 
    |> List.map (fun x -> System.Random.Shared.NextDouble(), x)
    |> List.sortBy fst
    |> List.map snd

// let take_max (l : int list) =
//     l
//     |> List.mapi (fun i x -> x, System.Random.Shared.NextDouble(), i)
//     |> List.max
//     |> fun (_,_,i) -> i

let take_max (l : int list) =
    l
    |> List.mapi (fun i x -> x, i)
    |> List.max
    |> fun (_,i) -> i

take_max [1;2;3;4;5;5;5;5;5;5]

