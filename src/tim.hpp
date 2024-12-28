#ifndef __NUCC_TIME_CHECK
#define __NUCC_TIME_CHECK

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

std::string getCurrentTime() {
    // Get the current time as a time_point
    auto now = std::chrono::system_clock::now();

    // Convert to time_t to get calendar time
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    // Convert to tm structure for formatting
    std::tm local_tm = *std::localtime(&now_c);

    // Get milliseconds
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    // Use stringstream for formatting
    std::stringstream ss;
    ss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << '.'
       << std::setfill('0') << std::setw(3) << milliseconds.count();

    return ss.str();
}

#endif // !__NUCC_TIME_CHECK
