__global__ void selection_k_radius_gpu(int b, int m, int k, float radius, const int* idx, const float* val, int* idx_out, float* val_out){
    int batch_index = blockIdx.x;
    int stride = batch_index * m * k;
    idx += stride;
    val += stride;
    idx_out += stride;
    val_out += stride;
    for(int i = threadIdx.x; i < m;i += blockDim.x) {

        for(int j = 0;j < k;j ++) {
            if(val[i * k + j] < radius) {
                idx_out[i * k + j] = idx[i * k + j];
                val_out[i * k + j] = val[i * k + j];
            } else {
                idx_out[i * k + j] = idx[i * k ];
                val_out[i * k + j] = val[i * k ];
            }
        }
    }
}

__global__ void cube_select(int b, int n,float radius, const float* xyz, int* idx_out) {
    int batch_idx = blockIdx.x;
    xyz += batch_idx * n * 3;
    idx_out += batch_idx * n * 8;
    float temp_dist[8];
    float judge_dist = radius * radius;
    for(int i = threadIdx.x; i < n;i += blockDim.x) {
        float x = xyz[i * 3];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        for(int j = 0;j < 8;j ++) {
            temp_dist[j] = 1e8;
            idx_out[i * 8 + j] = i; // if not found, just return itself..
        }
        for(int j = 0;j < n;j ++) {
            if(i == j) continue;
            float tx = xyz[j * 3];
            float ty = xyz[j * 3 + 1];
            float tz = xyz[j * 3 + 2];
            float dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
            if(dist > judge_dist) continue;
            int _x = (tx > x);
            int _y = (ty > y);
            int _z = (tz > z);
            int temp_idx = _x * 4 + _y * 2 + _z;
            if(dist < temp_dist[temp_idx]) {
                idx_out[i * 8 + temp_idx] = j;
                temp_dist[temp_idx] = dist;
            }
        }
    }
}

__global__ void cube_select_two(int b, int n,float radius, const float* xyz, int* idx_out) {
    int batch_idx = blockIdx.x;
    xyz += batch_idx * n * 3;
    idx_out += batch_idx * n * 16;
    float temp_dist[16];
    float judge_dist = radius * radius;
    for(int i = threadIdx.x; i < n;i += blockDim.x) {
        float x = xyz[i * 3];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        for(int j = 0;j < 16;j ++) {
            temp_dist[j] = judge_dist;
            idx_out[i * 16 + j] = i; // if not found, just return itself..
        }
        for(int j = 0;j < n;j ++) {
            if(i == j) continue;
            float tx = xyz[j * 3];
            float ty = xyz[j * 3 + 1];
            float tz = xyz[j * 3 + 2];
            float dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
            if(dist > judge_dist) continue;
            int _x = (tx > x);
            int _y = (ty > y);
            int _z = (tz > z);
            int temp_idx = _x * 8 + _y * 4 + _z * 2;
            bool flag = false;
            for(int k = 0;k < 2;k ++) {
                if (dist < temp_dist[temp_idx + k]) {
                    flag = true;
                }
                if (flag) {
                    for (int kk = 1; kk >= k + 1; kk --) {
                        idx_out[i * 16 + temp_idx + kk] = idx_out[i * 16 + temp_idx + kk - 1];
                        temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
                    }
                    idx_out[i * 16 + temp_idx + k] = j;
                    temp_dist[temp_idx + k] = dist;
                    break;
                }
            }

        }
    }
}

__global__ void cube_select_four(int b, int n,float radius, const float* xyz, int* idx_out) {
    int batch_idx = blockIdx.x;
    xyz += batch_idx * n * 3;
    idx_out += batch_idx * n * 32;
    float temp_dist[32];
    float judge_dist = radius * radius;
    for(int i = threadIdx.x; i < n;i += blockDim.x) {
        float x = xyz[i * 3];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        for(int j = 0;j < 32;j ++) {
            temp_dist[j] = judge_dist;
            idx_out[i * 32 + j] = i; // if not found, just return itself..
        }
        for(int j = 0;j < n;j ++) {
            if(i == j) continue;
            float tx = xyz[j * 3];
            float ty = xyz[j * 3 + 1];
            float tz = xyz[j * 3 + 2];
            float dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
            if(dist > judge_dist) continue;
            int _x = (tx > x);
            int _y = (ty > y);
            int _z = (tz > z);
            int temp_idx = _x * 16 + _y * 8 + _z * 4;
            bool flag = false;
            for(int k = 0;k < 4;k ++) {
                if (dist < temp_dist[temp_idx + k]) {
                    flag = true;
                }
                if (flag) {
                    for (int kk = 3; kk >= k + 1; kk --) {
                        idx_out[i * 32 + temp_idx + kk] = idx_out[i * 32 + temp_idx + kk - 1];
                        temp_dist[temp_idx + kk] = temp_dist[temp_idx + kk - 1];
                    }
                    idx_out[i * 32 + temp_idx + k] = j;
                    temp_dist[temp_idx + k] = dist;
                    break;
                }
            }

        }
    }
}



void selectionKRadiusLauncher(int b, int m, int k, float radius, const int* idx, const float* val, int* idx_out, float* val_out){
    selection_k_radius_gpu<<<b,256>>>(b, m, k, radius, idx, val, idx_out, val_out);
}
void cubeSelectLauncher(int b, int n, float radius, const float* xyz, int* idx_out) {
    cube_select<<<b, 512>>>(b, n, radius, xyz, idx_out);
}
void cubeSelectTwoLauncher(int b, int n, float radius, const float* xyz, int* idx_out) {
    cube_select_two<<<b, 512>>>(b, n, radius, xyz, idx_out);
}
void cubeSelectFourLauncher(int b, int n, float radius, const float* xyz, int* idx_out) {
    cube_select_four<<<b, 512>>>(b, n, radius, xyz, idx_out);
}
