module Spiral.Trading.Conditions

open System
open System.Net.Http
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading

[<CLIMutable>]
type UpdateRule = {
    [<JsonPropertyName("updates_high_low")>]
    UpdatesHighLow: bool

    [<JsonPropertyName("updates_open_close")>]
    UpdatesOpenClose: bool

    [<JsonPropertyName("updates_volume")>]
    UpdatesVolume: bool
}

[<CLIMutable>]
type UpdateRules = {
    [<JsonPropertyName("consolidated")>]
    Consolidated: UpdateRule option

    [<JsonPropertyName("market_center")>]
    MarketCenter: UpdateRule option
}

[<CLIMutable>]
type SipMapping = {
    [<JsonPropertyName("CTA")>]
    CTA: string option

    [<JsonPropertyName("UTP")>]
    UTP: string option

    [<JsonPropertyName("OPRA")>]
    OPRA: string option
}

[<CLIMutable>]
type Condition = {
    [<JsonPropertyName("id")>]
    Id: int

    [<JsonPropertyName("name")>]
    Name: string

    [<JsonPropertyName("description")>]
    Description: string option

    [<JsonPropertyName("asset_class")>]
    AssetClass: string

    [<JsonPropertyName("data_types")>]
    DataTypes: string[]

    [<JsonPropertyName("type")>]
    Type: string

    [<JsonPropertyName("sip_mapping")>]
    SipMapping: SipMapping option

    [<JsonPropertyName("update_rules")>]
    UpdateRules: UpdateRules option
}

[<CLIMutable>]
type ConditionsResponse = {
    [<JsonPropertyName("results")>]
    Results: Condition[] option

    [<JsonPropertyName("status")>]
    Status: string

    [<JsonPropertyName("request_id")>]
    RequestId: string

    [<JsonPropertyName("count")>]
    Count: int
}

let private baseUrl = "https://api.polygon.io/v3/reference/conditions"

let private jsonOptions =
    let options = JsonSerializerOptions()
    options.PropertyNameCaseInsensitive <- true
    options

/// Fetch trade conditions for stocks from the API
let fetchConditions
    (httpClient: HttpClient)
    (apiKey: string)
    (assetClass: string option)
    (dataType: string option)
    (ct: CancellationToken)
    : Async<Result<Condition[], string>> =
    async {
        let queryParams =
            [ assetClass |> Option.map (fun a -> $"asset_class={a}")
              dataType |> Option.map (fun d -> $"data_type={d}")
              Some "limit=1000"
              Some $"apiKey={apiKey}" ]
            |> List.choose id
            |> String.concat "&"

        let url = $"{baseUrl}?{queryParams}"

        try
            let! response = httpClient.GetAsync(url, ct) |> Async.AwaitTask
            response.EnsureSuccessStatusCode() |> ignore
            let! content = response.Content.ReadAsStringAsync(ct) |> Async.AwaitTask
            let parsed = JsonSerializer.Deserialize<ConditionsResponse>(content, jsonOptions)

            return Ok (parsed.Results |> Option.defaultValue [||])
        with
        | ex -> return Error ex.Message
    }

/// Print conditions in a readable table format
let printConditionsTable (conditions: Condition[]) =
    printfn ""
    printfn "%-4s  %-35s  %-32s  %-4s  %-4s  %-5s  %-5s  %-6s" "ID" "Name" "Type" "CTA" "UTP" "Hi/Lo" "Op/Cl" "Volume"
    printfn "%s" (String.replicate 110 "-")

    for c in conditions |> Array.sortBy (fun c -> c.Id) do
        let cta = c.SipMapping |> Option.bind (fun s -> s.CTA) |> Option.defaultValue "-"
        let utp = c.SipMapping |> Option.bind (fun s -> s.UTP) |> Option.defaultValue "-"

        let hiLo, opCl, vol =
            match c.UpdateRules |> Option.bind (fun r -> r.Consolidated) with
            | Some rule ->
                (if rule.UpdatesHighLow then "Y" else "N"),
                (if rule.UpdatesOpenClose then "Y" else "N"),
                (if rule.UpdatesVolume then "Y" else "N")
            | None -> "-", "-", "-"

        let name = if c.Name.Length > 35 then c.Name.Substring(0, 32) + "..." else c.Name
        printfn "%-4d  %-35s  %-32s  %-4s  %-4s  %-5s  %-5s  %-6s" c.Id name c.Type cta utp hiLo opCl vol

    printfn ""
    printfn "Total: %d conditions" conditions.Length
    printfn ""
    printfn "Legend: Hi/Lo = Updates High/Low, Op/Cl = Updates Open/Close, Volume = Updates Volume"
    printfn "        CTA = Consolidated Tape Association code, UTP = Unlisted Trading Privileges code"
