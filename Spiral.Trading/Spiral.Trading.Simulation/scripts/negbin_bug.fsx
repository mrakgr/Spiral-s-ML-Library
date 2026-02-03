#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)
let r = 100.0
let p = 0.5

let nb = NegativeBinomial(r, p, rng)

printfn "NegativeBinomial(r=%.0f, p=%.1f):" r p
printfn "  .Mean property: %.1f" nb.Mean
printfn ""

let samples = Array.init 10000 (fun _ -> nb.Sample())
printfn "Empirical from 10000 samples:"
printfn "  Mean: %.1f" (Array.averageBy float samples)
printfn "  Variance: %.1f" (let m = Array.averageBy float samples in samples |> Array.averageBy (fun x -> (float x - m) ** 2.0))
