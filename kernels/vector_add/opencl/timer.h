#ifndef TIMER_H_
#define TIMER_H_

#include <sys/time.h>

double clock()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double) tv.tv_usec * 1e-6 + (double)tv.tv_sec;
}

#endif
