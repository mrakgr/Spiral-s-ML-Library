#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// Test Poisson sampling speed for large lambda
printfn "Testing Poisson sampling speed..."

for lambda in [10.0; 100.0; 1000.0; 3000.0] do
    let sw = System.Diagnostics.Stopwatch.StartNew()
    let p = Poisson(lambda, rng)
    let sample = p.Sample()
    sw.Stop()
    printfn "lambda=%.0f: sample=%d, time=%dms" lambda sample sw.ElapsedMilliseconds
