#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

printfn "Testing Gamma sampling speed..."

for r in [10.0; 100.0; 1000.0; 3000.0] do
    let rate = 1.0
    let sw = System.Diagnostics.Stopwatch.StartNew()
    let g = Gamma(r, rate, rng)
    let sample = g.Sample()
    sw.Stop()
    printfn "r=%.0f: sample=%.1f, time=%dms" r sample sw.ElapsedMilliseconds
