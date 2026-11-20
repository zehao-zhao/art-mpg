
#pragma once
#include <atomic>
#include <chrono>
#include <iostream>
#include <array>
#include <vector>
namespace rb {
struct Counters {
 std::atomic<unsigned long long> joint_local_hits{0};
 std::atomic<unsigned long long> joint_local_misses{0};

 std::atomic<unsigned long long> joint_context_hits{0};
 std::atomic<unsigned long long> joint_context_misses{0};
 std::atomic<unsigned long long> joint_context_full{0};
 std::atomic<unsigned long long> joint_direction_hits{0};
 std::atomic<unsigned long long> joint_direction_misses{0};
 std::atomic<unsigned long long> joint_direction_full{0};

  std::atomic<unsigned long long> paths{0}, sample_ns{0}, network_ns{0}, network_api_calls{0}, forwards{0};
  std::atomic<unsigned long long> solves{0}, solver_ns{0}, solver_iterations{0}, retries{0}, chains{0};
  std::atomic<unsigned long long> inverse_weights{0}, inverse_cap{0}, contributions{0}, contribution_cap{0};
  std::atomic<unsigned long long> length_pdf_queries{0}, length_pdf_floor{0}, retry_max{0};
  std::array<std::atomic<unsigned long long>, 2051> histogram{};
};
inline Counters c;
inline float common_length_floor=1e-6f;
struct LocalCounters {
 unsigned long long joint_local_hits=0;
 unsigned long long joint_local_misses=0;

 unsigned long long joint_context_hits=0;
 unsigned long long joint_context_misses=0;
 unsigned long long joint_context_full=0;
 unsigned long long joint_direction_hits=0;
 unsigned long long joint_direction_misses=0;
 unsigned long long joint_direction_full=0;

 unsigned long long paths=0;
 unsigned long long sample_ns=0;
 unsigned long long network_ns=0;
 unsigned long long network_api_calls=0;
 unsigned long long forwards=0;
 unsigned long long solves=0;
 unsigned long long solver_ns=0;
 unsigned long long solver_iterations=0;
 unsigned long long retries=0;
 unsigned long long chains=0;
 unsigned long long inverse_weights=0;
 unsigned long long inverse_cap=0;
 unsigned long long contributions=0;
 unsigned long long contribution_cap=0;
 unsigned long long length_pdf_queries=0;
 unsigned long long length_pdf_floor=0;
 unsigned long long retry_max=0;
 std::array<unsigned long long,2051> histogram{};
 std::vector<size_t> touched;
};
inline thread_local LocalCounters lc;

struct Timer {
  std::atomic<unsigned long long> &target;
  std::chrono::steady_clock::time_point begin;
  Timer(std::atomic<unsigned long long>&t):target(t),begin(std::chrono::steady_clock::now()){}
  ~Timer(){target.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count(),std::memory_order_relaxed);}
};
inline void record_retries(unsigned long long n) {
  lc.chains++;lc.retries+=n;lc.retry_max=std::max(lc.retry_max,n);
  size_t bin=std::min(n,2050ull);
  if(lc.histogram[bin]==0)lc.touched.push_back(bin);
  lc.histogram[bin]++;
}
template<typename T> inline T floor_pdf(T low,T val) {
  low=T(common_length_floor);lc.length_pdf_queries++;if(val<low)lc.length_pdf_floor++;return std::max(low,val);
}
template<typename T> inline T checked_pdf(T val) { return floor_pdf(T(common_length_floor),val); }
inline void merge() {
 if(lc.joint_local_hits){c.joint_local_hits.fetch_add(lc.joint_local_hits,std::memory_order_relaxed);lc.joint_local_hits=0;}
 if(lc.joint_local_misses){c.joint_local_misses.fetch_add(lc.joint_local_misses,std::memory_order_relaxed);lc.joint_local_misses=0;}

 if(lc.joint_context_hits){c.joint_context_hits.fetch_add(lc.joint_context_hits,std::memory_order_relaxed);lc.joint_context_hits=0;}
 if(lc.joint_context_misses){c.joint_context_misses.fetch_add(lc.joint_context_misses,std::memory_order_relaxed);lc.joint_context_misses=0;}
 if(lc.joint_context_full){c.joint_context_full.fetch_add(lc.joint_context_full,std::memory_order_relaxed);lc.joint_context_full=0;}
 if(lc.joint_direction_hits){c.joint_direction_hits.fetch_add(lc.joint_direction_hits,std::memory_order_relaxed);lc.joint_direction_hits=0;}
 if(lc.joint_direction_misses){c.joint_direction_misses.fetch_add(lc.joint_direction_misses,std::memory_order_relaxed);lc.joint_direction_misses=0;}
 if(lc.joint_direction_full){c.joint_direction_full.fetch_add(lc.joint_direction_full,std::memory_order_relaxed);lc.joint_direction_full=0;}

 if(lc.paths){c.paths.fetch_add(lc.paths,std::memory_order_relaxed);lc.paths=0;}
 if(lc.sample_ns){c.sample_ns.fetch_add(lc.sample_ns,std::memory_order_relaxed);lc.sample_ns=0;}
 if(lc.network_ns){c.network_ns.fetch_add(lc.network_ns,std::memory_order_relaxed);lc.network_ns=0;}
 if(lc.network_api_calls){c.network_api_calls.fetch_add(lc.network_api_calls,std::memory_order_relaxed);lc.network_api_calls=0;}
 if(lc.forwards){c.forwards.fetch_add(lc.forwards,std::memory_order_relaxed);lc.forwards=0;}
 if(lc.solves){c.solves.fetch_add(lc.solves,std::memory_order_relaxed);lc.solves=0;}
 if(lc.solver_ns){c.solver_ns.fetch_add(lc.solver_ns,std::memory_order_relaxed);lc.solver_ns=0;}
 if(lc.solver_iterations){c.solver_iterations.fetch_add(lc.solver_iterations,std::memory_order_relaxed);lc.solver_iterations=0;}
 if(lc.retries){c.retries.fetch_add(lc.retries,std::memory_order_relaxed);lc.retries=0;}
 if(lc.chains){c.chains.fetch_add(lc.chains,std::memory_order_relaxed);lc.chains=0;}
 if(lc.inverse_weights){c.inverse_weights.fetch_add(lc.inverse_weights,std::memory_order_relaxed);lc.inverse_weights=0;}
 if(lc.inverse_cap){c.inverse_cap.fetch_add(lc.inverse_cap,std::memory_order_relaxed);lc.inverse_cap=0;}
 if(lc.contributions){c.contributions.fetch_add(lc.contributions,std::memory_order_relaxed);lc.contributions=0;}
 if(lc.contribution_cap){c.contribution_cap.fetch_add(lc.contribution_cap,std::memory_order_relaxed);lc.contribution_cap=0;}
 if(lc.length_pdf_queries){c.length_pdf_queries.fetch_add(lc.length_pdf_queries,std::memory_order_relaxed);lc.length_pdf_queries=0;}
 if(lc.length_pdf_floor){c.length_pdf_floor.fetch_add(lc.length_pdf_floor,std::memory_order_relaxed);lc.length_pdf_floor=0;}
 auto previous=c.retry_max.load(std::memory_order_relaxed);
 while(previous<lc.retry_max&&!c.retry_max.compare_exchange_weak(previous,lc.retry_max,std::memory_order_relaxed)){}
 lc.retry_max=0;
 for(size_t bin:lc.touched){c.histogram[bin].fetch_add(lc.histogram[bin],std::memory_order_relaxed);lc.histogram[bin]=0;}
 lc.touched.clear();
}
struct Reporter {
 ~Reporter(){
#define RB_PRINT(n) std::cout << "RENDER " #n " " << c.n.load() << std::endl;
 RB_PRINT(joint_context_hits) RB_PRINT(joint_context_misses) RB_PRINT(joint_context_full) RB_PRINT(joint_direction_hits) RB_PRINT(joint_direction_misses) RB_PRINT(joint_direction_full) 
 RB_PRINT(joint_local_hits) RB_PRINT(joint_local_misses) 
 RB_PRINT(paths) RB_PRINT(sample_ns) RB_PRINT(network_ns) RB_PRINT(network_api_calls) RB_PRINT(forwards)
 RB_PRINT(solves) RB_PRINT(solver_ns) RB_PRINT(solver_iterations) RB_PRINT(retries) RB_PRINT(chains)
 RB_PRINT(inverse_weights) RB_PRINT(inverse_cap) RB_PRINT(contributions) RB_PRINT(contribution_cap)
 RB_PRINT(length_pdf_queries) RB_PRINT(length_pdf_floor) RB_PRINT(retry_max)
 for(size_t i=0;i<c.histogram.size();++i) if(c.histogram[i].load())
   std::cout<<"RENDER retry_hist_"<<i<<" "<<c.histogram[i].load()<<std::endl;
#undef RB_PRINT
 }
};
inline Reporter reporter;
}
