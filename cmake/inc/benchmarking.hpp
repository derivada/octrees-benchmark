//
// Created by ruben.laso on 22/09/22.
//

#ifndef RULE_BASED_CLASSIFIER_CPP_BENCHMARKING_HPP
#define RULE_BASED_CLASSIFIER_CPP_BENCHMARKING_HPP

#include "TimeWatcher.hpp"
#include <algorithm>
#include <numeric>

namespace benchmarking
{

template<typename T = double>
class Stats
{
	private:
	std::vector<T> values_;

	T mean_{};

	mutable bool computed_median_{};
	mutable T    median_{};

	mutable bool computed_stdev_{};
	mutable T    stdev_{};

	public:
	inline void add_value(const T value)
	{
		mean_ = (mean_ * values_.size() + value) / (values_.size() + 1);
		values_.push_back(value);
	}

	inline auto values() const { return values_; }

	inline auto size() const { return values_.size(); }

	inline auto mean() const { return mean_; }

	inline auto accumulated() const { return std::accumulate(values_.begin(), values_.end(), T{}); }

	inline auto median() const
	{
		if (!computed_median_)
		{
			auto values = values_;
			std::sort(values.begin(), values.end());
			median_ = values.at(values.size() / 2);

			computed_median_ = true;
		}

		return median_;
	}

	inline auto stdev() const
	{
		if (!computed_stdev_)
		{
			std::vector<T> diff(values_.size());
			std::transform(values_.begin(), values_.end(), diff.begin(), std::bind2nd(std::minus<T>(), mean_));

			stdev_ = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
			stdev_ = std::sqrt(stdev_ / values_.size());

			computed_stdev_ = true;
		}

		return stdev_;
	}
};

template<typename F>
auto benchmark(const size_t repeats, F function)
{
	Stats stats;

	for (size_t i = 0; i < repeats; i++)
	{
		TimeWatcher tw;
		tw.start();
		function();
		tw.stop();
		stats.add_value(tw.getElapsedDecimalSeconds());
	}

	return stats;
}

template<typename F>
void benchmark(const std::string& description, const size_t repeats, F function, std::ostream& os = std::clog)
{
	const auto result = benchmark(repeats, function);

	os << "Benchmark \"" << description << "\". Repeats: " << repeats << ". Accumulated: " << result.accumulated()
	   << " s. Median: " << result.median() << " s. Mean: " << result.mean() << " s. Stdev: " << result.stdev() << "."
	   << '\n';
}
} // namespace benchmarking

#endif /* end of include guard: RULE_BASED_CLASSIFIER_CPP_BENCHMARKING_HPP */
