#pragma once
#include <papi.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <utility>

constexpr int CACHE_EVENTS[] = {
    PAPI_L1_DCM,
    PAPI_L2_DCM,
    PAPI_L3_TCM,
};

constexpr std::pair<const char*, const char*> NATIVE_EVENTS[] = {
    // Add native events here if needed
};

constexpr int NUM_NATIVE_EVENTS = sizeof(NATIVE_EVENTS) / sizeof(NATIVE_EVENTS[0]);

std::pair<std::vector<int>, std::vector<std::string>> buildCombinedEventList();

void printPapiResults(const std::vector<int>& events,
                      const std::vector<std::string>& descs,
                      const std::vector<long long>& values);

int initPapiEventSet(std::vector<int>& events);