module Spiral.Trading.Config

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

/// Internal JSON representation matching the api_key.json format
[<CLIMutable>]
type private ConfigJson = {
    [<JsonPropertyName("massive_api_key")>]
    MassiveApiKey: string

    [<JsonPropertyName("massive_s3_access_key")>]
    MassiveS3AccessKey: string

    [<JsonPropertyName("massive_s3_secret_key")>]
    MassiveS3SecretKey: string
}

/// Load Massive API configuration from a JSON file
let loadConfig (path: string) : Result<MassiveConfig, string> =
    if not (File.Exists path) then
        Error $"Config file not found: {path}"
    else
        try
            let json = File.ReadAllText path
            let options = JsonSerializerOptions()
            options.Converters.Add(JsonFSharpConverter())

            let config = JsonSerializer.Deserialize<ConfigJson>(json, options)

            if String.IsNullOrWhiteSpace config.MassiveApiKey then
                Error "massive_api_key is missing or empty"
            elif String.IsNullOrWhiteSpace config.MassiveS3AccessKey then
                Error "massive_s3_access_key is missing or empty"
            elif String.IsNullOrWhiteSpace config.MassiveS3SecretKey then
                Error "massive_s3_secret_key is missing or empty"
            else
                Ok {
                    ApiKey = config.MassiveApiKey
                    S3AccessKey = config.MassiveS3AccessKey
                    S3SecretKey = config.MassiveS3SecretKey
                }
        with ex ->
            Error $"Failed to parse config: {ex.Message}"

/// Load config or throw exception (for CLI convenience)
let loadConfigOrFail (path: string) : MassiveConfig =
    match loadConfig path with
    | Ok config -> config
    | Error msg -> failwith msg
