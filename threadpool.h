#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "macros.h"

class TaskList;
class ThreadPool;

class WorkerThread {
public:
  WorkerThread() {}

  ~WorkerThread() {
    running_ = false;
    thread_.join();
  }

  void Start(ThreadPool * threadPool) {
    threadPool_ = threadPool;
    thread_ = std::thread(&WorkerThread::Work, this);
  }

private:
  void Work();

private:
  std::thread thread_;
  std::atomic_bool running_ = true;
  ThreadPool * threadPool_;
  std::unique_ptr<std::function<void()>> job_;
};

class ThreadPool {
public:
  ThreadPool(std::size_t size) : size_(size) {
    threads_.reset(new WorkerThread[size_]);
    for (std::size_t i = 0u; i < size_; ++i) {
      threads_[i].Start(this);
    }
  }

  struct Job {
    TaskList * list;
    std::function<void()> job;
  };

  void DoJob(TaskList * taskList, std::function<void()> job) {
    {
      std::unique_lock<std::mutex> lock(lock_);
      jobs_.push(Job{taskList, job});
    }
    jobAdded_.notify_one();
  }

  bool GetJob(Job & job) {
    std::unique_lock<std::mutex> lock(lock_);

    if (jobs_.empty())
      jobAdded_.wait(lock);

    if (jobs_.empty())
      return false;

    job = jobs_.front();
    jobs_.pop();
    return true;
  }

  std::size_t Size() const { return size_; }

private:
  std::size_t size_;
  std::unique_ptr<WorkerThread[]> threads_;
  std::mutex lock_;
  std::queue<Job> jobs_;
  std::condition_variable jobAdded_;
};

class TaskList {
public:
  TaskList(ThreadPool & threadPool)
      : threadPool_(threadPool), completed_(0) {
    taskDone_ = taskPromise_.get_future();
  }

  void AddTask(std::function<void()> func) {
    tasks_.push_back(func);
  }

  void Run() {
    for (auto && task : tasks_)
      threadPool_.DoJob(this, task);

    taskDone_.wait();
  }

  void TaskFinished() {
    {
      // Make sure the mutex is freed by the time we destruct
      std::unique_lock<std::mutex> lock(lock_);
      if (++completed_ != tasks_.size())
        return;
    }
    taskPromise_.set_value();
  }

private:
  ThreadPool & threadPool_;
  int completed_;
  std::mutex lock_;
  std::promise<void> taskPromise_;
  std::future<void> taskDone_;
  std::vector<std::function<void()>> tasks_;
};

ThreadPool * GetCpuSizedThreadPool();
