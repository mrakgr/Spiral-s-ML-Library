#r "nuget: MathNet.Numerics"

open MathNet.Numerics.Distributions

let r, p = 100.0, 0.5
let nb = NegativeBinomial(r, p)

printfn "NegativeBinomial(r=%.0f, p=%.1f)" r p
printfn "  .Mean property: %.1f" nb.Mean
printfn "  Empirical mean (10k samples): %.1f" 
    (Array.init 10000 (fun _ -> nb.Sample()) |> Array.averageBy float)
