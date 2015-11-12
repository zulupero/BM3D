#ifndef UTIL_H
#define UTIL_H

#include <time.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

class Timer
{
    public:
        static float getElapsedTime();
        static float getElapsedTimeTotal();
        static void start();
        static void startCuda();
        static void startTotal();
        static void stopTotal();
        static void stop();
        static void stopCuda();
        static void show(const char* prefix);
        static void showResults();
        static void add(const char* prefix);
        static void addCuda(const char* prefix);
        static void addTotal(const char* prefix);

    private:

        class TimerResult
        {
            public:
                enum type_t 
                {
                    CPU,
                    CUDA
                };

                TimerResult()
                {}

                ~TimerResult()
                {}

                const char* prefix;
                float elapsedTime;
                type_t type;
        };
        static clock_t startTime;
        static clock_t stopTime;
        static clock_t startTotalTime;
        static clock_t stopTotalTime;

        static cudaEvent_t startCudaTimer;
        static cudaEvent_t stopCudaTimer;

        static vector<TimerResult> results;
};

#endif // BM3D_H
