#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "benchmarking_stats.hpp"


namespace gb_wrappers {

inline void initializeGoogleBenchmark() {
    static std::once_flag benchmarkInitFlag;
    std::call_once(benchmarkInitFlag, []() {
        static char arg0[] = "octrees-benchmark";
        static char* argv[] = { arg0 };
        int argc = 1;
        benchmark::Initialize(&argc, argv);
    });
}

class ContextOnceReporter final : public benchmark::BenchmarkReporter {
    public:
        explicit ContextOnceReporter(benchmark::BenchmarkReporter* baseReporter,
            double* capturedMilliseconds = nullptr,
            std::vector<double>* capturedIterationMilliseconds = nullptr,
            bool captureRealTime = false)
            : baseReporter_(baseReporter),
              capturedMilliseconds_(capturedMilliseconds),
              capturedIterationMilliseconds_(capturedIterationMilliseconds),
              captureRealTime_(captureRealTime) {}

        bool ReportContext(const Context& context) override {
            static std::atomic<bool> contextAlreadyPrinted{false};
            if (!contextAlreadyPrinted.exchange(true)) {
                return baseReporter_->ReportContext(context);
            }
            return true;
        }

        void ReportRuns(const std::vector<Run>& reports) override {
            auto formattedReports = reports;
            for (const auto& run : reports) {
                if (run.run_type == Run::RunType::RT_Iteration) {
                    const double value = captureRealTime_ ? run.GetAdjustedRealTime()
                                                          : run.GetAdjustedCPUTime();
                    if (capturedMilliseconds_ != nullptr) {
                        *capturedMilliseconds_ = value;
                    }
                    if (capturedIterationMilliseconds_ != nullptr) {
                        capturedIterationMilliseconds_->push_back(value);
                    }
                }
            }
            for (auto& run : formattedReports) {
                run.run_name.min_warmup_time.clear();
                run.run_name.iterations.clear();
            }
            baseReporter_->ReportRuns(formattedReports);
        }

        void Finalize() override {
            baseReporter_->Finalize();
        }

    private:
        benchmark::BenchmarkReporter* baseReporter_;
        double* capturedMilliseconds_;
        std::vector<double>* capturedIterationMilliseconds_;
        bool captureRealTime_;
};

inline benchmarking::BenchmarkingStats<double> statsFromIterationMilliseconds(
    const std::vector<double>& iterationMilliseconds,
    bool warmupEnabled) {
    benchmarking::BenchmarkingStats<double> stats(warmupEnabled);
    for (double valueMs : iterationMilliseconds) {
        stats.add_value(valueMs / 1000.0);
    }
    if (iterationMilliseconds.empty()) {
        stats.add_value(0.0);
    }
    if (warmupEnabled) {
        stats.set_warmup_value(0.0);
    }
    return stats;
}

}  // namespace gb_wrappers
