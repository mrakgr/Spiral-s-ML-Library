#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

let dispExp = 0.0
let p = Math.Pow(2.0, -dispExp)
let rate = 50.0
let duration = 60.0
let mean = rate * duration
let r = mean * p / (1.0 - p)
let gammaRate = p / (1.0 - p)

printfn "dispExp=%.1f, p=%.3f, r=%.1f, gammaRate=%.3f" dispExp p r gammaRate

let sw = System.Diagnostics.Stopwatch.StartNew()
let lambda = Gamma(r, gammaRate, rng).Sample()
printfn "Gamma sample: %.1f (took %dms)" lambda sw.ElapsedMilliseconds

sw.Restart()
let sample = Poisson(lambda, rng).Sample()
printfn "Poisson sample: %d (took %dms)" sample sw.ElapsedMilliseconds
