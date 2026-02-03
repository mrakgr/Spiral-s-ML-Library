#r "nuget: MathNet.Numerics"

open System

let medianPerSec = 50.0
let meanPerSec = 60.0

printfn "=== LogNormal Parameters vs Duration ==="
printfn ""
printfn "duration | medianCount | meanCount | mu     | sigma"
printfn "---------|-------------|-----------|--------|------"

for duration in [10.0; 30.0; 60.0; 120.0] do
    let medianCount = medianPerSec * duration
    let meanCount = meanPerSec * duration
    let mu = log(medianCount)
    let sigma = sqrt(2.0 * log(meanCount / medianCount))
    printfn "%7.0fs | %11.0f | %9.0f | %6.3f | %.3f" duration medianCount meanCount mu sigma
