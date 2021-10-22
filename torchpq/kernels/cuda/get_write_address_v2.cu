#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif


typedef long long ll_t;
typedef unsigned char uint8_t;
extern "C"
__global__ void get_write_address(
  const uint8_t* __restrict__ isEmpty,
  const ll_t* __restrict__ divStart,
  const ll_t* __restrict__ divSize,
  const ll_t* __restrict__ labels,
  const ll_t* __restrict__ ioa,
  ll_t* __restrict__ write_adr,
  int n_slots, int n_labels
) {
  int tid = threadIdx.x; // thread ID
  int lid = blockIdx.x * _TPB_ + tid; // label ID

  if (lid < n_labels){
    const ll_t threadLabel = labels[lid];
    const ll_t threadIoa = ioa[lid];
    const ll_t threadDivStart = divStart[threadLabel];
    const ll_t threadDivSize = divSize[threadLabel];

    ll_t counter = 0;
    for (int i=0; i<threadDivSize; i++){
      ll_t adr = threadDivStart + i;
      if (adr < n_slots){
        uint8_t empty = isEmpty[adr];
        if (empty == 1){
          if (counter == threadIoa){
            write_adr[lid] = adr;
            break;
          }
          counter ++;
        }
      }
    } // end for i
  }
}