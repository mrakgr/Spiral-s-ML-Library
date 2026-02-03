#r "nuget: MathNet.Numerics"

open MathNet.Numerics.Distributions

// Poisson hangs when lambda = infinity
let p = Poisson(infinity)
printfn "Attempting to sample Poisson(infinity)..."
let sample = p.Sample()  // This hangs forever
printfn "Sample: %d" sample
