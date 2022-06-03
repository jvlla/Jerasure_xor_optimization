#include <stdio.h>
#include <time.h>
#include "timing.h"
#define LIMIT 100

double timing_delta(struct timespec * t1, struct timespec * t2)
{
  static double timing_time = 0;
  
  if (timing_time == 0) {
    // first time call timing_delta(), calculate the time spend by clock_gettime()
    struct timespec time_out_begin, time_out_end, time_in_begin, time_in_end;
    for (int i = 0; i < LIMIT; ++i)
    {
        clock_gettime(CLOCK_REALTIME, &time_out_begin);
        clock_gettime(CLOCK_REALTIME, &time_in_begin);
        clock_gettime(CLOCK_REALTIME, &time_in_end);
        clock_gettime(CLOCK_REALTIME, &time_out_end);
        timing_time += (double)(time_out_end.tv_nsec - time_out_begin.tv_nsec);
    }
    timing_time /= LIMIT;
  }

  return (double)t2->tv_sec - t1->tv_sec + (t2->tv_nsec - t1->tv_nsec - timing_time) / 1e9;
}
