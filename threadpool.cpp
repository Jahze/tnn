#include "threadpool.h"

ThreadPool * GetCpuSizedThreadPool() {
  static ThreadPool * pool = nullptr;

  if (!pool)
    pool = new ThreadPool(std::thread::hardware_concurrency());

  return pool;
}

void WorkerThread::Work() {
  SetFloatingPointExceptionMode();

  while (running_) {
    ThreadPool::Job job;

    if (threadPool_->GetJob(job)) {
      job.job();
      job.list->TaskFinished();
    }
  }
}
