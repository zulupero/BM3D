#include "util.h"
#include <stdio.h>

clock_t Timer::stopTime = 0;
clock_t Timer::startTime = 0;
vector<Timer::TimerResult> Timer::results;

float Timer::getElapsedTime()
{
    return (((float)stopTime) - ((float)startTime));
}

void Timer::start()
{
    startTime = clock();
}

void Timer::stop()
{
    stopTime = clock();
}

void Timer::show(const char* prefix)
{
    printf("\nExecution of %s: %f ms", prefix, getElapsedTime());
}

void Timer::showResults()
{
    for(unsigned int i=0; i< results.size(); ++i)
    {
        Timer::TimerResult result = results[i];
        printf("\nExecution of %s: %f s", result.prefix, (result.elapsedTime/CLOCKS_PER_SEC));
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

