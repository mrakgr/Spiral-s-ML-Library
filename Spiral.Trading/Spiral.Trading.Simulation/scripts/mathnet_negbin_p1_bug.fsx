#r "nuget: MathNet.Numerics"

open MathNet.Numerics.Distributions

// NegativeBinomial hangs when p = 1.0
let nb = NegativeBinomial(10.0, 1.0)
printfn "Attempting to sample NegativeBinomial(r=10, p=1.0)..."
let sample = nb.Sample()  // This hangs forever
printfn "Sample: %d" sample
