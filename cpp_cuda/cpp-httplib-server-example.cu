#include "nlohmann/json.hpp"
#include "httplib.h"

int main() {
    // HTTP
    httplib::Server svr;
    
    svr.Get("/hi", [](const httplib::Request &, httplib::Response &res) {
      nlohmann::json j{"name", "example"};



      // j["name"] = "example";
      // j["value"] = 123;
      // j["items"] = { "one", "two", "three" };

      std::string json_str = j.dump();
      res.set_content(json_str, "application/json");
    });
    
    svr.listen("0.0.0.0", 8080);
}
