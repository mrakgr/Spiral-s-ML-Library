#include <xoshiro.h>
#include <random>
#include <map>
#include <iomanip>

int main()
{
    xso::rng gen;

    std::uint32_t value = gen();

    std::cout << value << '\n';

    // // lambda: draws a sample from a normal distribution and rounds it to an integer.
    // std::normal_distribution dist{5.0, 2.0};
    // auto random_int = [&dist, &gen] { return int(std::round(dist(gen))); };

    // // Run many trials and create a histogram of the results in integer buckets.
    // std::map<int, int> hist{};
    // for (int n = 0; n != 25'000; ++n) ++hist[random_int()];

    // // Print that histogram.
    // for (auto [bucket, count] : hist) {
    //     auto n = static_cast<std::size_t>(count / 100);
    //     std::cout << std::setw(2) << bucket << ' ' << std::string(n, '*') << '\n';
    // }
}