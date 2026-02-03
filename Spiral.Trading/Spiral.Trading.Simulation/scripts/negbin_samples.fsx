#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)
let mean = 100.0
let p = 0.5
let r = mean * p / (1.0 - p)

printfn "NegBin: mean=%.0f, p=%.1f, r=%.1f" mean p r
printfn ""

let nb = NegativeBinomial(r, p, rng)
let samples = Array.init 50 (fun _ -> nb.Sample())

printfn "50 samples:"
for i in 0 .. 4 do
    let row = samples.[i*10 .. i*10+9] |> Array.map (sprintf "%4d") |> String.concat " "
    printfn "%s" row

printfn ""
printfn "Stats from 10000 samples:"
let bigSample = Array.init 10000 (fun _ -> nb.Sample())
let sorted = Array.sort bigSample
printfn "  Mean: %.1f" (Array.averageBy float bigSample)
printfn "  Median: %d" sorted.[5000]
printfn "  Min: %d, Max: %d" (Array.min bigSample) (Array.max bigSample)
