void * jerasure_schedule_decode_lazy_thread(void * arg)
{
  struct decode_arg * p_decode_arg = (struct decode_arg *) arg;
  int i, tdone;

  // int flag = p_decode_arg->schedule[i][2];
  // printf("flag = %d\n", flag);
  // for (i = 1; p_decode_arg->schedule[i][0] >= 0; i++)
  //   if (p_decode_arg->schedule[i][2] != flag)
  //     printf("wrong here, flag = %d, curr = %d\n", flag, p_decode_arg->schedule[i][2]);

  while (true) {
    pthread_mutex_lock(p_decode_arg->pmutex);
    while(!p_decode_arg->ready)
      pthread_cond_wait(p_decode_arg->pfill, p_decode_arg->pmutex);
    p_decode_arg->ready = false;
    pthread_cond_signal(p_decode_arg->pempty);
    // do encoding  
    if (p_decode_arg->id == 0) {  
    for (tdone = 0; tdone < size_global; tdone += packetsize_global*w_global) {
      printf("%llx\n", p_decode_arg->ptrs[0]);
      jerasure_do_scheduled_operations(p_decode_arg->ptrs, p_decode_arg->schedule, packetsize_global);
      for (i = 0; i < k_global+m_global; i++) p_decode_arg->ptrs[i] += (packetsize_global*w_global);
    }
    printf("\n");
}
    pthread_mutex_unlock(p_decode_arg->pmutex);
  }
}

int jerasure_schedule_decode_lazy(int k, int m, int w, int *bitmatrix, int *erasures,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize, 
                            int smart)
{
  int i, j;
  static int **schedule;  // 这个要释放 暂时static一下
  static int * prev_erasures = NULL, * prev_bitmatrix = NULL;
  static int ***sub_schedules = NULL;
  static int sub_num;
  int *sub_counts;
  static pthread_cond_t empty[MAX_CPUS], fill[MAX_CPUS];
  static pthread_mutex_t mutex[MAX_CPUS];
  static struct decode_arg decode_args[MAX_CPUS];
  static pthread_t p[MAX_CPUS];
  int rc;
  // cpu_set_t mask;

  if (prev_erasures != erasures || prev_bitmatrix != bitmatrix) {
    // 先处理之前的，prev erase bitmatrix 要变 
    // 空间要释放 sub_schedules

    schedule = jerasure_generate_decoding_schedule(k, m, w, bitmatrix, erasures, smart);
    if (schedule == NULL) {
      return -1;
    }
    for (sub_num = 0; erasures[sub_num] >= 0; sub_num++)
      continue;
    sub_schedules = talloc(int **, sub_num);
    sub_counts = talloc(int, sub_num);
    for (i = 0; i < sub_num; i++) {
      sub_schedules[i] = talloc(int *, k*w*w+1);
      sub_counts[i] = 0;
    }
    for (i = 0; schedule[i][0] >= 0; i++)
      sub_schedules[schedule[i][2]-k][sub_counts[schedule[i][2]-k]++] = schedule[i];
    for (i = 0; i < sub_num; i++) {
      sub_schedules[i][sub_counts[i]] = talloc(int, 5);
      sub_schedules[i][sub_counts[i]][0] = -1;
    }

    k_global = k;
    m_global = m;
    w_global = w;
    size_global = size;
    packetsize_global = packetsize;
    for (i = 0; i < sub_num; i++)
    {
      pthread_cond_init(&empty[i], NULL);
      pthread_cond_init(&fill[i], NULL);
      pthread_mutex_init(&mutex[i], NULL);
      decode_args[i].pempty = &empty[i];
      decode_args[i].pfill = &fill[i];
      decode_args[i].ready = false;
      decode_args[i].pmutex = &mutex[i];
      rc = pthread_create(&p[i], NULL, jerasure_schedule_decode_lazy_thread, &decode_args[i]); assert(rc == 0);
      // CPU_ZERO(&mask);
      // CPU_SET(i, &mask);
      // assert(!pthread_setaffinity_np(p[i], sizeof(cpu_set_t), &mask));
      decode_args[i].schedule = sub_schedules[i];
      decode_args[i].id = i; // 这是是要删掉的-----------------------------------------------
    }

    prev_erasures = erasures;
    prev_bitmatrix = bitmatrix;
  }
  for (i = 0; i < sub_num; i++) {
    // 先这样，为了效率回头再改
    decode_args[i].ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
  } 
  // decode_args[0].ptrs = set_up_ptrs_for_scheduled_decoding(k, m, erasures, data_ptrs, coding_ptrs);
  // decode_args[0].schedule = schedule;

  for (i = 0; schedule[i][0] >= 0; i++)
    printf("%d %d %d %d %d\n", schedule[i][0], schedule[i][1], schedule[i][2]
      , schedule[i][3], schedule[i][4]);
  exit(-1);

  // printf("%x %x\n", sub_schedules[1], decode_args[1].schedule);

  printf("sub_count: %d %d\n", sub_counts[0], sub_counts[1]);
  for (i = 0; i < sub_num; i++) {
    for (j = 0; decode_args[i].schedule[j][0] >= 0; j++)
      printf("id = %d, %d %d %d %d %d\n", i, decode_args[i].schedule[j][0], decode_args[i].schedule[j][1]
        , decode_args[i].schedule[j][2], decode_args[i].schedule[j][3], decode_args[i].schedule[j][4]);
    // printf("\n\n");
    printf("\n\n");
  }
  exit(-1);


  for (i = 0; i < sub_num; i++)
  {
    pthread_mutex_lock(&mutex[i]);
    decode_args[i].ready = true;
    pthread_cond_signal(&fill[i]);
    pthread_mutex_unlock(&mutex[i]);
  }
  for (i = 0; i < sub_num; i++)
  {
    pthread_mutex_lock(&mutex[i]);
    while (decode_args[i].ready)
      pthread_cond_wait(&empty[i], &mutex[i]);
    pthread_mutex_unlock(&mutex[i]);
  }

  // for (tdone = 0; tdone < size; tdone += packetsize*w) {
  // jerasure_do_scheduled_operations(ptrs, schedule, packetsize);
  //   for (i = 0; i < k+m; i++) ptrs[i] += (packetsize*w);
  // }

  // jerasure_free_schedule(schedule);
  // free(ptrs);

  return 0;
}