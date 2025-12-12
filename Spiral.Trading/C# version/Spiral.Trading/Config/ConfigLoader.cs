using Newtonsoft.Json.Linq;
using System.IO;

namespace Spiral.Trading.Config
{
    public static class ConfigLoader
    {
        public static (string PolygonApiKey, string S3AccessKey, string S3SecretKey) LoadKeys(string path = "api_key.json")
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Config file not found: {path}");
            }

            var json = File.ReadAllText(path);
            var obj = JObject.Parse(json);

            var polygonKey = obj["massive_api_key"]?.ToString() ?? throw new InvalidOperationException("massive_api_key not found");
            var s3Access = obj["massive_s3_access_key"]?.ToString() ?? throw new InvalidOperationException("massive_s3_access_key not found");
            var s3Secret = obj["massive_s3_secret_key"]?.ToString() ?? throw new InvalidOperationException("massive_s3_secret_key not found");

            return (polygonKey, s3Access, s3Secret);
        }
    }
}
