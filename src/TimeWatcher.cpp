//
// Created by miguelyermo on 16/11/20.
//

#include "TimeWatcher.hpp"

#include <iomanip>
#include <memory>
#include <sstream>


// *** PUBLIC METHODS *** //
// ********************** //
void TimeWatcher::start()
{
	tStart = std::make_unique<std::chrono::high_resolution_clock::time_point>(

	    std::chrono::high_resolution_clock::now());
}
void TimeWatcher::stop()
{
	tEnd = std::make_unique<std::chrono::high_resolution_clock::time_point>(

	    std::chrono::high_resolution_clock::now());
}

std::shared_ptr<std::chrono::high_resolution_clock::duration> TimeWatcher::getElapsedTime()
{
	if (hasNulls()) return nullptr;
	return std::make_shared<std::chrono::high_resolution_clock::duration>(*tEnd - *tStart);
}

double TimeWatcher::getElapsedDecimalSeconds() { return ((double) getElapsedNanos()) / 1e9; }

long TimeWatcher::getElapsedSeconds()
{
	if (hasNulls()) return 0L;
	return std::chrono::duration_cast<std::chrono::seconds>(*getElapsedTime()).count();
}

long TimeWatcher::getElapsedMillis()
{
	if (hasNulls()) return 0L;
	return std::chrono::duration_cast<std::chrono::milliseconds>(*getElapsedTime()).count();
}

long TimeWatcher::getElapsedMicros()
{
  if (hasNulls()) return 0L;
  return std::chrono::duration_cast<std::chrono::microseconds>(*getElapsedTime()).count();
}

long TimeWatcher::getElapsedNanos()
{
	if (hasNulls()) return 0L;
	return std::chrono::duration_cast<std::chrono::nanoseconds>(*getElapsedTime()).count();
}

std::string TimeWatcher::getElapsedFormat()
{
	long seconds = getElapsedSeconds();
	long minutes = seconds / 60;
	long hours   = minutes / 60;
	minutes      = minutes % 60;
	seconds      = seconds % 60;
	std::stringstream strm;
	strm << hours << ":";
	strm << std::setfill('0') << std::setw(2);
	strm << minutes << ":" << seconds;
	return strm.str();
}

void TimeWatcher::reportSeconds(ostream& os, const string& msg)
{
	std::stringstream ss;
	ss << msg << getElapsedSeconds();
	os << ss.str();
}

void TimeWatcher::reportMillis(ostream& os, const string& msg)
{
	std::stringstream ss;
	ss << msg << getElapsedMillis() << std::endl;
	os << ss.str();
}

void TimeWatcher::reportFormat(ostream& os, const std::string& msg)
{
	std::stringstream ss;
	ss << msg << getElapsedFormat() << std::endl;
	os << ss.str();
}

// *** PRIVATE METHODS *** //
// ********************** //
bool TimeWatcher::hasNulls() { return tStart == nullptr || tEnd == nullptr; }
