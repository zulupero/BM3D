#ifndef UTIL_H
#define UTIL_H

#include <time.h>
#include <vector>
using namespace std;

class Timer
{
    public:
        static float getElapsedTime();
        static float getElapsedTimeTotal();
        static void start();
        static void startTotal();
        static void stopTotal();
        static void stop();
        static void show(const char* prefix);
        static void showResults();
        static void add(const char* prefix);
        static void addTotal(const char* prefix);

    private:

        class TimerResult
        {
            public:
                TimerResult()
                {}

                ~TimerResult()
                {}

                const char* prefix;
                float elapsedTime;
        };
        static clock_t startTime;
        static clock_t stopTime;
        static clock_t startTotalTime;
        static clock_t stopTotalTime;

        static vector<TimerResult> results;
};

#endif // BM3D_H
