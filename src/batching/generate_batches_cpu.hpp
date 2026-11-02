#pragma once

#include "router_state.hpp"

namespace cudb {

vector<vector<int>> generate_batches_rsmt(RouterState& state, vector<int> &nets2route, int MAX_BATCH_SIZE = 1000000,
                                          int sparsity_override = 0) {
    auto& X = state.device.X;
    auto& Y = state.device.Y;
    auto& nets = state.nets;
    if (sparsity_override < 0) throw std::invalid_argument("sparsity must be positive, or 0 for the original default");
    int sparsity = (mode == 1 || mode == 2) ? 5 : 20;
    if (sparsity_override > 0) {
        sparsity = sparsity_override;
        printf("BATCH_SPARSITY backend=cpu mode=%d value=%d\n", mode, sparsity);
    }
    auto _time = elapsed_time();
    vector<vector<int>> batches;
    vector<vector<bool>> batch_vis;
    // vector<int> lookupTable(X * Y, 0);
   
    auto has_conflict = [&] (int net_id, int batch_id) {

        for(auto p : nets[net_id].points) 
        {
            assert(p < X * Y);
            if(batch_vis[batch_id][p])
            {
                return true;
            }
        }
        return false;
    };

    auto mark_3x3 = [&] (int pos, int batch_id) {
        int _x = pos / Y, _y = pos % Y;
        for(int x = _x - 1; x <= _x + 1; x++) if(0 <= x && x < X)
            for(int y = _y - 1; y < _y + 1; y++) if(0 <= y && y < Y)
                if(x == _x || y == _y) batch_vis[batch_id][x * Y + y] = 1;
    };


    long long segment_len = 0, segment_cnt = 0, failed = 0;
    for(auto net_id : nets2route) {
        int batch_id = -1;
        for(int i = 0; i < batches.size(); i++) if(batches[i].size() < MAX_BATCH_SIZE)
        {
            if(!has_conflict(net_id, i)) { batch_id = i; break; }
            else failed++;
        }
        if(batch_id == -1) {
            batch_id = batches.size();
            batches.emplace_back(vector<int> ());
            batch_vis.emplace_back(vector<bool> (X * Y, 0));
        }
        batches[batch_id].emplace_back(net_id);
        for(auto seg : nets[net_id].rsmt_h_segments) {
            segment_len += seg.second / Y - seg.first / Y;
            segment_cnt++;
            for(auto p = seg.first; p <= seg.second; p += sparsity * Y) batch_vis[batch_id][p] = 1;
        }
        for(auto seg : nets[net_id].rsmt_v_segments) {
            for(auto p = seg.first; p <= seg.second; p += sparsity) batch_vis[batch_id][p] = 1;
        }
        for(auto p : nets[net_id].points) mark_3x3(p, batch_id);
    }
    _time = elapsed_time() - _time;
    printf("INFO: AVG LEN = %lld/%lld = %.2f\n", segment_len,segment_cnt, 1.0 * segment_len / segment_cnt);
    printf("INFO: failed = %lld\n", failed);
    if(LOG) cout << setw(40) << "Batch" << setw(20) << "#Nets" << setw(20) << "#Batches" << setw(20) << "Time" << endl;
    if(LOG) cout << setw(40) << "Generation" << setw(20) << nets2route.size() << setw(20) << batches.size() << setw(20) << setprecision(2) << _time << endl;
    return move(batches);
}

} // namespace cudb
