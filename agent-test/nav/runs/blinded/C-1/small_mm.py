"""Small-row BF16 projection, with exact FP32 matrix accumulation.

Specialized to 16 input rows and K=1024. BF16 inputs are expanded to FP32
in threadgroup memory; results round to BF16 at the projection boundary.
"""
import mlx.core as mx

source=r'''
uint tid=thread_position_in_threadgroup.x;
uint sg=simdgroup_index_in_threadgroup;
uint m0=threadgroup_position_in_grid.y*16;
uint n0=threadgroup_position_in_grid.x*NT;
uint K=w_shape[1], N=w_shape[0];
threadgroup float xs[16*BK];
threadgroup float ws[NT*BK];
threadgroup float os[16*NT];
simdgroup_float8x8 acc[NT/16];
for(int j=0;j<NT/16;++j) acc[j]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
uint gm=sg/2, gn=sg%2;
for(uint kk=0;kk<K;kk+=BK) {
    for(uint i=tid;i<16*BK;i+=128) xs[i]=float(x[(m0+i/BK)*K+kk+i%BK]);
    for(uint i=tid;i<NT*BK;i+=128) ws[i]=float(w[(n0+i/BK)*K+kk+i%BK]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=0;t<BK;t+=8) {
        simdgroup_float8x8 a;
        simdgroup_load(a,xs+gm*8*BK+t,BK,ulong2(0),false);
        for(uint j=0;j<NT/16;++j) {
            simdgroup_float8x8 b;
            simdgroup_load(b,ws+(gn*(NT/2)+j*8)*BK+t,BK,ulong2(0),true);
            simdgroup_multiply_accumulate(acc[j],a,b,acc[j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint j=0;j<NT/16;++j)simdgroup_store(acc[j],os+gm*8*NT+gn*(NT/2)+j*8,NT,ulong2(0),false);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<16*NT;i+=128)y[(m0+i/NT)*N+n0+i%NT]=T(os[i]);
'''
kernel=mx.fast.metal_kernel(name='small_bf16_mm2',input_names=['x','w'],output_names=['y'],
    header='#include <metal_simdgroup_matrix>\n',source=source)
def matmul(x,w,nt=32):
    bk=16
    M=x.size//x.shape[-1];N=w.shape[0]
    return kernel(inputs=[x,w],template=[('T',x.dtype),('BK',bk),('NT',nt)],grid=(N//nt*128,M//16,1),
        threadgroup=(128,1,1),output_shapes=[(*x.shape[:-1],N)],output_dtypes=[x.dtype])[0]
