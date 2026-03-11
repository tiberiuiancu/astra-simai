/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "entry.h"
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
// simulon bridge headers (paths relative to csrc/ include root)
#include "astra_bindings/ns3_runner.hh"
#include "astra_bindings/topology_bridge.hh"
#include "astra_bindings/workload_bridge.hh"
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#ifdef NS3_MPI
#include "ns3/mpi-interface.h"
#include <mpi.h>
#endif

#define RESULT_PATH "./ncclFlowModel_"

using namespace std;
using namespace ns3;

extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
extern uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
extern GPUType gpu_type;
extern std::vector<int>NVswitchs;

struct sim_event {
  void *buffer;
  uint64_t count;
  int type;
  int dst;
  int tag;
  string fnType;
};

class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
private:
  int npu_offset;

public:
  queue<sim_event> sim_event_queue;
  ASTRASimNetwork(int rank, int npu_offset) : AstraNetworkAPI(rank) {
    this->npu_offset = npu_offset;
  }
  ~ASTRASimNetwork() {}
  int sim_comm_size(AstraSim::sim_comm comm, int *size) { return 0; }
  int sim_finish() {
    // Do not call exit() — this runs inside a Python .so library.
    return 0;
  }
  double sim_time_resolution() { return 0; }
  int sim_init(AstraSim::AstraMemoryAPI *MEM) { return 0; }
  AstraSim::timespec_t sim_get_time() {
    AstraSim::timespec_t timeSpec;
    timeSpec.time_val = Simulator::Now().GetNanoSeconds();
    return timeSpec;
  }
  virtual void sim_schedule(AstraSim::timespec_t delta,
                            void (*fun_ptr)(void *fun_arg), void *fun_arg) {
    task1 t;
    t.type = 2;
    t.fun_arg = fun_arg;
    t.msg_handler = fun_ptr;
    t.schTime = delta.time_val;
    Simulator::Schedule(NanoSeconds(t.schTime), t.msg_handler, t.fun_arg);
    return;
  }
  virtual int sim_send(void *buffer,   
                       uint64_t count, 
                       int type,       
                       int dst,
                       int tag,                       
                       AstraSim::sim_request *request, 
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    dst += npu_offset;
    task1 t;
    t.src = rank;
    t.dest = dst;
    t.count = count;
    t.type = 0;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    {
      #ifdef NS3_MTP
      MtpInterface::explicitCriticalSection cs;
      #endif
      sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
    }
    SendFlow(rank, dst, count, msg_handler, fun_arg, tag, request);
    return 0;
  }
  virtual int sim_recv(void *buffer, uint64_t count, int type, int src, int tag,
                       AstraSim::sim_request *request,
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    AstraSim::ncclFlowTag flowTag = request->flowTag;
    src += npu_offset;
    task1 t;
    t.src = src;
    t.dest = rank;
    t.count = count;
    t.type = 1;
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
    AstraSim::EventType event = ehd->event;
    tag = ehd->flowTag.tag_id;
    NcclLog->writeLog(NcclLogLevel::DEBUG,"[Receive event registration] src %d sim_recv on rank %d tag_id %d channdl id %d",src,rank,tag,ehd->flowTag.channel_id);
    
    if (recvHash.find(make_pair(tag, make_pair(t.src, t.dest))) !=
        recvHash.end()) {
      uint64_t count = recvHash[make_pair(tag, make_pair(t.src, t.dest))];
      if (count == t.count) {
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
        assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
        if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
          AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
          receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
          ehd->flowTag = pending_tag;
        } 
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else if (count > t.count) {
        recvHash[make_pair(tag, make_pair(t.src, t.dest))] = count - t.count;
        assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
        if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
          AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
          receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
          ehd->flowTag = pending_tag;
        } 
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else {
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
        t.count -= count;
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
      }
    } else {
      if (expeRecvHash.find(make_pair(tag, make_pair(t.src, t.dest))) ==
          expeRecvHash.end()) {
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
          NcclLog->writeLog(NcclLogLevel::DEBUG," [Packet arrived late, registering first] recvHash do not find expeRecvHash.new make src  %d dest  %d t.count:  %llu channel_id  %d current_flow_id  %d",t.src,t.dest,t.count,tag,flowTag.current_flow_id);
          
      } else {
        uint64_t expecount =
            expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))].count;
          NcclLog->writeLog(NcclLogLevel::DEBUG," [Packet arrived late, re-registering] recvHash do not find expeRecvHash.add make src  %d dest  %d expecount:  %d t.count:  %d tag_id  %d current_flow_id  %d",t.src,t.dest,expecount,t.count,tag,flowTag.current_flow_id);
          
      }
    }
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif

sim_recv_end_section:    
    return 0;
  }
  void handleEvent(int dst, int cnt) {
  }
};

struct user_param {
  int thread;
  string workload;
  string network_topo;
  string network_conf;
  user_param() {
    thread = 1;
    workload = "";
    network_topo = "";
    network_conf = "";
  };
  ~user_param(){};
};

static int user_param_prase(int argc,char * argv[],struct user_param* user_param){
  int opt;
  while ((opt = getopt(argc,argv,"ht:w:g:s:n:c:"))!=-1){
    switch (opt)
    {
    case 'h':
      /* code */
      std::cout<<"-t    number of threads,default 1"<<std::endl;
      std::cout<<"-w    workloads default none "<<std::endl;
      std::cout<<"-n    network topo"<<std::endl;
      std::cout<<"-c    network_conf"<<std::endl;
      return 1;
      break;
    case 't':
      user_param->thread = stoi(optarg);
      break;
    case 'w':
      user_param->workload = optarg;
      break;
    case 'n':
      user_param->network_topo = optarg;
      break;
    case 'c':
      user_param->network_conf = optarg;
      break;
    default:
      std::cerr<<"-h    help message"<<std::endl;
      return 1;
    }
  }
  return 0 ;
}

int main(int argc, char *argv[]) {
  struct user_param user_param;
  MockNcclLog::set_log_name("SimAI.log");
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::INFO," init SimAI.log ");
  if(user_param_prase(argc,argv,&user_param)){
    return 0;
  }
  #ifdef NS3_MTP
  MtpInterface::Enable(user_param.thread);
  #endif
  
  main1(user_param.network_topo,user_param.network_conf);
  int nodes_num = node_num - switch_num;
  int gpu_num = node_num - nvswitch_num - switch_num;

  std::map<int, int> node2nvswitch; 
  for(int i = 0; i < gpu_num; ++ i) {
    node2nvswitch[i] = gpu_num + i / gpus_per_server;
  }
  for(int i = gpu_num; i < gpu_num + nvswitch_num; ++ i){
    node2nvswitch[i] = i;
    NVswitchs.push_back(i);
  } 

  LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
  LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
  LogComponentEnable("GENERIC_SIMULATION", LOG_LEVEL_INFO);

  std::vector<ASTRASimNetwork *> networks(nodes_num, nullptr);
  std::vector<AstraSim::Sys *> systems(nodes_num, nullptr);

  for (int j = 0; j < nodes_num; j++) {
    networks[j] =
        new ASTRASimNetwork(j ,0);
    systems[j ] = new AstraSim::Sys(
        networks[j], 
        nullptr,                  
        j,                        
        0,               
        1,                        
        {nodes_num},        
        {1},          
        "", 
        user_param.workload, 
        1, 
        1,          
        1,          
        1,
        0,                 
        RESULT_PATH, 
        "test1",            
        true,               
        false,               
        gpu_type,
        {gpu_num},
        NVswitchs,
        gpus_per_server
    );
    systems[j ]->nvswitch_id = node2nvswitch[j];
    systems[j ]->num_gpus = nodes_num - nvswitch_num;
  }
  for (int i = 0; i < nodes_num; i++) {
    systems[i]->workload->fire();
  }
  std::cout << "simulator run " << std::endl;

  Simulator::Run();
  Simulator::Stop(Seconds(2000000000));
  Simulator::Destroy();
  
  #ifdef NS3_MPI
  MpiInterface::Disable ();
  #endif
  return 0;
}

// ---------------------------------------------------------------------------
// simulon::astra::run_ns3 — in-process NS3 simulation entry point
// Defined here so it shares ASTRASimNetwork class and NS3 globals.
// ---------------------------------------------------------------------------

namespace simulon {
namespace astra {

static std::string ns3_write_temp_file(const std::string& content,
                                        const std::string& suffix) {
    char buf[256];
    snprintf(buf, sizeof(buf), "/tmp/simulon_ns3_XXXXXX%s", suffix.c_str());
    int fd = mkstemps(buf, static_cast<int>(suffix.size()));
    if (fd < 0) throw std::runtime_error("Cannot create temp file for NS3");
    ::write(fd, content.data(), content.size());
    ::close(fd);
    return std::string(buf);
}

static std::string make_ns3_conf(const NetworkTopology& /*topo*/,
                                  const WorkloadTrace& /*wl*/) {
    return "CC_MODE 3\nENABLE_QCN 1\nUSE_DYNAMIC_PFC_THRESHOLD 1\n"
           "PAUSE_TIME 5\nPACKET_PAYLOAD_SIZE 1024\n"
           "L2_CHUNK_SIZE 4000\nL2_ACK_INTERVAL 156\nL2_BACK_TO_ZERO 0\n"
           "BUFFER_SIZE 600\nSIMULATOR_STOP_TIME 10000000.0\nENABLE_TRACE 0\n"
           "DATA_RATE 100Gbps\nLINK_DELAY 1us\n"
           "RATE_AI 50Mbps\nRATE_HAI 100Mbps\nMIN_RATE 100Mbps\n"
           "DCTCP_RATE_AI 1000Mbps\nERROR_RATE_PER_LINK 0\n"
           "FLOW_FILE /dev/null\nTRACE_FILE /dev/null\n"
           "TRACE_OUTPUT_FILE /dev/null\nFCT_OUTPUT_FILE /dev/null\n"
           "PFC_OUTPUT_FILE /dev/null\nSEND_OUTPUT_FILE /dev/null\n"
           "QLEN_MON_FILE /dev/null\nBW_MON_FILE /dev/null\n"
           "RATE_MON_FILE /dev/null\nCNP_MON_FILE /dev/null\n";
}

AnalyticalResults run_ns3(
    const NetworkTopology& topology,
    const WorkloadTrace& workload
) {
    AnalyticalResults results;
    results.success = false;
    results.total_time_ns = 0.0;
    results.compute_time_ns = 0.0;
    results.communication_time_ns = 0.0;
    results.completed_layers = 0;

    std::string topo_path, conf_path;
    try {
        NetWorkParam net_param = toNetWorkParam(topology);
        UserParam::getInstance()->net_work_param = net_param;

        topo_path = ns3_write_temp_file(toTopoFileContent(topology), ".txt");
        conf_path = ns3_write_temp_file(make_ns3_conf(topology, workload), ".txt");

        if (main1(topo_path, conf_path) != 0) {
            results.error_message = "NS3 network initialization failed";
            ::unlink(topo_path.c_str());
            ::unlink(conf_path.c_str());
            return results;
        }

        // After main1(): node_num, switch_num, nvswitch_num, gpus_per_server set.
        int nodes_num = static_cast<int>(node_num - switch_num);   // GPUs + NVSwitches
        int gpu_num   = static_cast<int>(node_num - nvswitch_num - switch_num);  // GPUs only

        // Populate NVswitchs vector (mirrors the logic in original main())
        NVswitchs.clear();
        for (int i = gpu_num; i < gpu_num + static_cast<int>(nvswitch_num); ++i) {
            NVswitchs.push_back(i);
        }

        // Build node→nvswitch map
        std::map<int, int> node2nvswitch;
        for (int i = 0; i < gpu_num; ++i) {
            node2nvswitch[i] = gpu_num + i / static_cast<int>(gpus_per_server);
        }
        for (int i = gpu_num; i < gpu_num + static_cast<int>(nvswitch_num); ++i) {
            node2nvswitch[i] = i;
        }

        std::vector<int> physical_dims  = {nodes_num};
        std::vector<int> queues_per_dim = {1};

        std::vector<ASTRASimNetwork*> networks(nodes_num, nullptr);
        std::vector<AstraSim::Sys*>   systems(nodes_num, nullptr);

        for (int j = 0; j < nodes_num; ++j) {
            networks[j] = new ASTRASimNetwork(j, 0);
            systems[j]  = new AstraSim::Sys(
                networks[j],
                nullptr,
                j,
                0,
                1,
                physical_dims,
                queues_per_dim,
                "",
                "",          // workload path empty — we use DIRECT_INIT below
                1.0, 1.0, 1.0,
                1, 0,
                "",          // no output path
                "",          // no run name
                false,       // separate_log disabled
                false,       // rendezvous disabled
                net_param.gpu_type,
                {gpu_num},
                NVswitchs,
                static_cast<int>(gpus_per_server),
                "DIRECT_INIT"
            );
            systems[j]->nvswitch_id = node2nvswitch[j];
            systems[j]->num_gpus    = gpu_num;
        }

        bool all_ok = true;
        for (int j = 0; j < nodes_num; ++j) {
            initialize_workload_direct(systems[j]->workload, systems[j], workload);
            if (!systems[j]->workload || !systems[j]->workload->initialized) {
                all_ok = false;
                break;
            }
        }

        if (!all_ok) {
            results.error_message = "Failed to initialize workload for all nodes";
        } else {
            for (int j = 0; j < nodes_num; ++j) {
                systems[j]->workload->fire();
            }

            Simulator::Run();
            results.total_time_ns    = static_cast<double>(
                Simulator::Now().GetNanoSeconds());
            results.completed_layers = workload.num_layers;
            results.success          = true;
            Simulator::Destroy();
        }

    } catch (const std::exception& e) {
        results.error_message = std::string("NS3 simulation error: ") + e.what();
    } catch (...) {
        results.error_message = "Unknown NS3 simulation error";
    }

    if (!topo_path.empty()) ::unlink(topo_path.c_str());
    if (!conf_path.empty()) ::unlink(conf_path.c_str());
    return results;
}

}  // namespace astra
}  // namespace simulon
