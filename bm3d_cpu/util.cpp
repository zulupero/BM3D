#include "util.h"
#include <stdio.h>

clock_t Timer::stopTime = 0;
clock_t Timer::startTime = 0;
clock_t Timer::stopTotalTime = 0;
clock_t Timer::startTotalTime = 0;
vector<Timer::TimerResult> Timer::results;

float Timer::getElapsedTime()
{
    return (((float)stopTime) - ((float)startTime));
}

float Timer::getElapsedTimeTotal()
{
    return (((float)stopTotalTime) - ((float)startTotalTime));
}

void Timer::start()
{
    startTime = clock();
}

void Timer::startTotal()
{
    startTotalTime = clock();
}

void Timer::stop()
{
    stopTime = clock();
}

void Timer::stopTotal()
{
    stopTotalTime = clock();
}

void Timer::show(const char* prefix)
{
    printf("\n%s: %f ms", prefix, getElapsedTime());
}

void Timer::showResults()
{
    printf("\n\nStatistics:");
    for(unsigned int i=0; i< results.size(); ++i)
    {
        Timer::TimerResult result = results[i];
        printf("\n%s: %f ms", result.prefix, (result.elapsedTime/CLOCKS_PER_SEC * 1000));
    }
}

void Timer::add(const char* prefix)
{
    stop();
    Timer::TimerResult result;
    result.prefix = prefix;
    result.elapsedTime = getElapsedTime();
    results.push_back(result);
}

void Timer::addTotal(const char* prefix)
{
    stopTotal();
    Timer::TimerResult result;
    result.prefix = prefix;
    result.elapsedTime = getElapsedTimeTotal();
    results.push_back(result);
}

