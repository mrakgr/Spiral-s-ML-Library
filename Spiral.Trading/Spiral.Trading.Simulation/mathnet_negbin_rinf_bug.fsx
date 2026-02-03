#r "nuget: MathNet.Numerics"

open MathNet.Numerics.Distributions

// NegativeBinomial hangs when r = infinity
let nb = NegativeBinomial(infinity, 0.5)
printfn "Attempting to sample NegativeBinomial(r=infinity, p=0.5)..."
let sample = nb.Sample()  // This hangs forever
printfn "Sample: %d" sample
