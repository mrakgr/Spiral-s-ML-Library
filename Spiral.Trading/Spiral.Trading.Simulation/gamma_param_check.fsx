#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

// MathNet Gamma(shape, rate) has mean = shape/rate
// But some parameterizations use Gamma(shape, scale) with mean = shape*scale

let shape = 10.0
let rateOrScale = 2.0

let g = Gamma(shape, rateOrScale)
printfn "Gamma(%.1f, %.1f):" shape rateOrScale
printfn "  .Mean property: %.1f" g.Mean
printfn "  If shape/rate: %.1f" (shape / rateOrScale)
printfn "  If shape*scale: %.1f" (shape * rateOrScale)
