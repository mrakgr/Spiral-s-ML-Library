#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

/// Build stopping probabilities from any CDF
/// cdf: cumulative distribution function F(x) = P(X <= x)
/// maxSteps: maximum number of discrete steps
let buildStoppingProbabilities (cdf: float -> float) (maxSteps: int) =
    [| for k in 0 .. maxSteps - 1 do
        let F_k1 = cdf (float (k + 1))    // CDF at t+1
        let F_k = cdf (float k)            // CDF at t
        let survival = 1.0 - F_k
        if survival > 1e-10 then
            (F_k1 - F_k) / survival
        else
            1.0  // force stop if we've exceeded reasonable range
    |]

/// Build stopping probabilities for log-normal distribution
let buildLogNormalStoppingProbabilities (mu: float) (sigma: float) (maxSteps: int) =
    let dist = LogNormal(mu, sigma)
    buildStoppingProbabilities dist.CumulativeDistribution maxSteps

/// Build stopping probabilities for normal distribution (truncated at 0)
let buildNormalStoppingProbabilities (mean: float) (stdDev: float) (maxSteps: int) =
    let dist = Normal(mean, stdDev)
    buildStoppingProbabilities dist.CumulativeDistribution maxSteps

let sampleSteps (stoppingProbs: float[]) (rng: Random) =
    let rec loop step =
        if step >= stoppingProbs.Length then
            stoppingProbs.Length  // cap at max
        elif rng.NextDouble() < stoppingProbs.[step] then
            step + 1  // stop here (return 1-based duration)
        else
            loop (step + 1)  // continue
    loop 0

// Test: morning session ends around 60 minutes with stddev 20
let morningProbs = buildNormalStoppingProbabilities 60.0 20.0 120
let rng = Random(42)

printfn "Stopping probabilities at key times:"
printfn "  t=30: %.4f" morningProbs.[29]
printfn "  t=50: %.4f" morningProbs.[49]
printfn "  t=60: %.4f" morningProbs.[59]
printfn "  t=70: %.4f" morningProbs.[69]
printfn "  t=90: %.4f" morningProbs.[89]

printfn "\nSampling 10 morning end times:"
for _ in 1..10 do
    let endTime = sampleSteps morningProbs rng
    printfn "  %d minutes" endTime