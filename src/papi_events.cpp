#include "benchmarking/papi_events.hpp"

std::pair<std::vector<int>, std::vector<std::string>> buildCombinedEventList() {
    std::vector<int> fullEventList;
    std::vector<std::string> fullEventListDescs;
    PAPI_event_info_t info;

    for (auto ev : CACHE_EVENTS) {
        fullEventList.push_back(ev);
        if (PAPI_get_event_info(ev, &info) == PAPI_OK) {
            fullEventListDescs.emplace_back(info.short_descr);
        } else {
            fullEventListDescs.emplace_back("Unknown PAPI event");
        }
    }

    for (int i = 0; i < NUM_NATIVE_EVENTS; ++i) {
        int code;
        if (PAPI_event_name_to_code(const_cast<char*>(NATIVE_EVENTS[i].first), &code) == PAPI_OK) {
            fullEventList.push_back(code);
            fullEventListDescs.push_back(NATIVE_EVENTS[i].second);
        } else {
            std::cerr << "Failed to convert native event name: " << NATIVE_EVENTS[i].first << "\n";
        }
    }

    return {fullEventList, fullEventListDescs};
}

void printPapiResults(const std::vector<int>& events,
                      const std::vector<std::string>& descs,
                      const std::vector<long long>& values) {
    std::ios_base::fmtflags originalFlags = std::cout.flags();
    std::streamsize originalPrecision = std::cout.precision();

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "PAPI Performance Counters:\n";
    for (size_t i = 0; i < events.size(); ++i) {
        std::cout << "  " << descs[i] << ": "
                  << static_cast<double>(values[i]) << "\n";
    }

    std::cout.flags(originalFlags);
    std::cout.precision(originalPrecision);
}

int initPapiEventSet(std::vector<int>& events) {
    int eventSet = PAPI_NULL;

    if (PAPI_create_eventset(&eventSet) != PAPI_OK) {
        std::cerr << "Failed to create PAPI event set." << std::endl;
        return PAPI_NULL;
    }

    for (int i = 0; i < events.size(); ++i) {
        if (PAPI_add_event(eventSet, events[i]) != PAPI_OK) {
            std::cerr << "Failed to add event " << events[i] << " at index " << i << std::endl;
            PAPI_cleanup_eventset(eventSet);
            PAPI_destroy_eventset(&eventSet);
            return PAPI_NULL;
        }
    }

    return eventSet;
}
