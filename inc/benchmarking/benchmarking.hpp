//
// Created by ruben.laso on 22/09/22.
//

#ifndef RULE_BASED_CLASSIFIER_CPP_BENCHMARKING_HPP
#define RULE_BASED_CLASSIFIER_CPP_BENCHMARKING_HPP

#include "TimeWatcher.hpp"
#include <algorithm>
#include <numeric>
#include <functional>
#include <optional>
#include <papi.h>

namespace benchmarking
{

template<typename T = double>
class Stats
{
	private:
	std::vector<T> values_;
	T warmupValue_;

	T mean_{};

	mutable bool computed_median_{};
	mutable T    median_{};

	mutable bool computed_stdev_{};
	mutable T    stdev_{};

	const bool warmup;

	public:
	Stats(bool warmup): warmup(warmup) {}
	
	inline void set_warmup_value(const T value) { warmupValue_ = value; }

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
			std::transform(values_.begin(), values_.end(), diff.begin(), std::bind(std::minus<T>(), std::placeholders::_1, mean_));

			stdev_ = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
			stdev_ = std::sqrt(stdev_ / values_.size());

			computed_stdev_ = true;
		}

		return stdev_;
	}

	inline const T warmupValue() const { return warmupValue_; }

	inline const bool usedWarmup() const { return warmup; }
};

template<typename F>
auto benchmark(const size_t repeats, F function, bool warmup = true, int eventSet = PAPI_NULL, long long *eventValues = nullptr)
{
    Stats<double> stats(warmup);
	size_t totalRepeats = repeats + warmup;
	for (size_t i = 0; i < totalRepeats; i++)
	{
		TimeWatcher tw;
		
		// IMPORTANT: cache failures are only measured on the last repeat, when cache is the warmest
		if(eventSet != PAPI_NULL && eventValues != nullptr && i == totalRepeats-1) {
			if (PAPI_start(eventSet) != PAPI_OK) {
				std::cout << "Failed to start PAPI." << std::endl;
				exit(1);
			}
		}

		tw.start();
		function();
		tw.stop();

		// IMPORTANT: cache failures are only measured on the last repeat, when cache is the warmest
		if(eventSet != PAPI_NULL && eventValues != nullptr && i == totalRepeats-1) {
			if (PAPI_stop(eventSet, eventValues) != PAPI_OK) {
				std::cout << "Failed to stop PAPI." << std::endl;
				exit(1);
			}
		}


		if(warmup && i == 0)
			stats.set_warmup_value(tw.getElapsedDecimalSeconds());
		else
			stats.add_value(tw.getElapsedDecimalSeconds());
	}
	return stats;
}

// General case for non-void return type
template <typename ReturnType, typename F>
auto benchmark(const size_t repeats, F function, bool warmup = true, int eventSet = PAPI_NULL, long long *eventValues = nullptr)
{
	Stats<double> stats(warmup);  // Store time in double for simplicity
	size_t totalRepeats = repeats + warmup;
	ReturnType returnValue;
	for (size_t i = 0; i < totalRepeats; i++)
	{
		TimeWatcher tw;

		// IMPORTANT: cache failures are only measured on the last repeat, when cache is the warmest
		if(eventSet != PAPI_NULL && eventValues != nullptr && i == totalRepeats-1) {
			if (PAPI_start(eventSet) != PAPI_OK) {
				std::cout << "Failed to start PAPI." << std::endl;
				exit(1);
			}
		}

		tw.start();
		returnValue = function();  // Capture the return value
		tw.stop();

		// IMPORTANT: cache failures are only measured on the last repeat, when cache is the warmest
		if(eventSet != PAPI_NULL && eventValues != nullptr && i == totalRepeats-1) {
			if (PAPI_stop(eventSet, eventValues) != PAPI_OK) {
				std::cout << "Failed to stop PAPI." << std::endl;
				exit(1);
			}
		}

		if (warmup && i == 0)
			stats.set_warmup_value(tw.getElapsedDecimalSeconds());
		else
			stats.add_value(tw.getElapsedDecimalSeconds());
	}

	return std::make_pair(stats, returnValue);  // Return both stats and result
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
