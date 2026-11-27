#include "kernel_operator.h"
using namespace AscendC;
// Each intermediate cast preserves the BF16 rounding of the original primitives.
extern "C" __global__ __aicore__ void fused_pointwise(GM_ADDR x,GM_ADDR style,GM_ADDR y,GM_ADDR out,uint32_t rows,uint32_t cols,uint32_t mode) {
    TPipe pipe;TBuf<TPosition::VECCALC> bx,bs,by,bout,bf,bg,bh;
    pipe.InitBuffer(bx,8192);pipe.InitBuffer(bs,8192);pipe.InitBuffer(by,8192);pipe.InitBuffer(bout,8192);
    pipe.InitBuffer(bf,16384);pipe.InitBuffer(bg,16384);pipe.InitBuffer(bh,16384);
    auto lx=bx.Get<bfloat16_t>();auto ls=bs.Get<bfloat16_t>();auto ly=by.Get<bfloat16_t>();auto lo=bout.Get<bfloat16_t>();
    auto f=bf.Get<float>();auto g=bg.Get<float>();auto h=bh.Get<float>();
    GlobalTensor<bfloat16_t> gx,gs,gy,go;gx.SetGlobalBuffer((__gm__ bfloat16_t*)x);gs.SetGlobalBuffer((__gm__ bfloat16_t*)style);gy.SetGlobalBuffer((__gm__ bfloat16_t*)y);go.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    for(uint32_t row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
        uint32_t off=row*cols;
        DataCopy(lx,gx[off],cols);DataCopy(ls,gs[2*cols],cols);DataCopy(ly,gy[off],cols);
        PipeBarrier<PIPE_ALL>();Cast(f,lx,RoundMode::CAST_NONE,cols);Cast(g,ls,RoundMode::CAST_NONE,cols);
        PipeBarrier<PIPE_V>();Mul(h,f,g,cols);PipeBarrier<PIPE_V>();Cast(lo,h,RoundMode::CAST_RINT,cols);
        PipeBarrier<PIPE_V>();Cast(h,lo,RoundMode::CAST_NONE,cols);Cast(g,ly,RoundMode::CAST_NONE,cols);
        PipeBarrier<PIPE_V>();Add(f,h,g,cols);PipeBarrier<PIPE_V>();Cast(lo,f,RoundMode::CAST_RINT,cols);
        PipeBarrier<PIPE_ALL>();DataCopy(go[off],lo,cols);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" __global__ __aicore__ void fused_qkv_rope(GM_ADDR x,GM_ADDR cos,GM_ADDR sin,GM_ADDR q,GM_ADDR k,GM_ADDR v,uint32_t rows,uint32_t qh,uint32_t kh,uint32_t dim,uint32_t rotate) {
    TPipe pipe;TBuf<TPosition::VECCALC> bx,bc,bs,bo,bf,bg,bh,bi;
    pipe.InitBuffer(bx,8192);pipe.InitBuffer(bc,512);pipe.InitBuffer(bs,512);pipe.InitBuffer(bo,8192);
    pipe.InitBuffer(bf,16384);pipe.InitBuffer(bg,1024);pipe.InitBuffer(bh,1024);pipe.InitBuffer(bi,16384);
    auto lx=bx.Get<bfloat16_t>();auto lc=bc.Get<bfloat16_t>();auto ls=bs.Get<bfloat16_t>();auto lo=bo.Get<bfloat16_t>();
    auto f=bf.Get<float>();auto c=bg.Get<float>();auto s=bh.Get<float>();auto tmp=bi.Get<float>();
    GlobalTensor<bfloat16_t> gx,gc,gs,gq,gk,gv;gx.SetGlobalBuffer((__gm__ bfloat16_t*)x);gc.SetGlobalBuffer((__gm__ bfloat16_t*)cos);gs.SetGlobalBuffer((__gm__ bfloat16_t*)sin);gq.SetGlobalBuffer((__gm__ bfloat16_t*)q);gk.SetGlobalBuffer((__gm__ bfloat16_t*)k);gv.SetGlobalBuffer((__gm__ bfloat16_t*)v);
    uint32_t heads=qh+2*kh,width=heads*dim;
    for(uint32_t row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
        DataCopy(lx,gx[row*width],width);DataCopy(lc,gc[row*dim],dim);DataCopy(ls,gs[row*dim],dim);PipeBarrier<PIPE_ALL>();
        Cast(f,lx,RoundMode::CAST_NONE,width);Cast(c,lc,RoundMode::CAST_NONE,dim);Cast(s,ls,RoundMode::CAST_NONE,dim);PipeBarrier<PIPE_V>();
        for(uint32_t h=0;h<qh+kh;h++) Mul(tmp[h*dim],f[h*dim],c,dim);
        PipeBarrier<PIPE_V>();
        for(uint32_t h=0;h<qh+kh;h++) {Mul(f[h*dim],f[h*dim],s,dim);}
        PipeBarrier<PIPE_V>();
        for(uint32_t h=0;h<qh+kh;h++) {Sub(tmp[h*dim],tmp[h*dim],f[h*dim+dim/2],dim/2);Add(tmp[h*dim+dim/2],tmp[h*dim+dim/2],f[h*dim],dim/2);}
        PipeBarrier<PIPE_V>();Cast(lo,tmp,RoundMode::CAST_RINT,(qh+kh)*dim);Cast(lo[(qh+kh)*dim],f[(qh+kh)*dim],RoundMode::CAST_RINT,kh*dim);PipeBarrier<PIPE_ALL>();
        DataCopy(gq[row*qh*dim],lo,qh*dim);DataCopy(gk[row*kh*dim],lo[qh*dim],kh*dim);DataCopy(gv[row*kh*dim],lo[(qh+kh)*dim],kh*dim);PipeBarrier<PIPE_ALL>();
    }
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" int launch_pointwise(void* stream,void* x,void* style,void* y,void* out,uint32_t rows,uint32_t cols,uint32_t mode) {fused_pointwise<<<24,nullptr,stream>>>(x,style,y,out,rows,cols,mode);return 0;}
extern "C" int launch_rope(void* stream,void* x,void* cos,void* sin,void* q,void* k,void* v,uint32_t rows,uint32_t qh,uint32_t kh,uint32_t dim,uint32_t rotate) {fused_qkv_rope<<<24,nullptr,stream>>>(x,cos,sin,q,k,v,rows,qh,kh,dim,rotate);return 0;}
#endif
extern "C" __global__ __aicore__ void fused_adarms(GM_ADDR x,GM_ADDR style,GM_ADDR out,uint32_t rows,uint32_t cols,float eps) {
    TPipe pipe;TBuf<TPosition::VECCALC> bx,bs,bo,bf,bg,bh,br;
    pipe.InitBuffer(bx,2048);pipe.InitBuffer(bs,2048);pipe.InitBuffer(bo,2048);pipe.InitBuffer(bf,4096);pipe.InitBuffer(bg,4096);pipe.InitBuffer(bh,4096);pipe.InitBuffer(br,4096);
    auto lx=bx.Get<bfloat16_t>();auto ls=bs.Get<bfloat16_t>();auto lo=bo.Get<bfloat16_t>();auto f=bf.Get<float>();auto g=bg.Get<float>();auto h=bh.Get<float>();auto r=br.Get<float>();
    GlobalTensor<bfloat16_t> gx,gs,go;gx.SetGlobalBuffer((__gm__ bfloat16_t*)x);gs.SetGlobalBuffer((__gm__ bfloat16_t*)style);go.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    for(uint32_t row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
        DataCopy(lx,gx[row*cols],cols);DataCopy(ls,gs,cols);PipeBarrier<PIPE_ALL>();
        Cast(f,lx,RoundMode::CAST_NONE,cols);Cast(g,ls,RoundMode::CAST_NONE,cols);PipeBarrier<PIPE_V>();
        Mul(h,f,f,cols);PipeBarrier<PIPE_V>();ReduceSum(r,h,h,cols);PipeBarrier<PIPE_V>();
        float denom=r.GetValue(0)/cols+eps;float inv=1.0f/sqrt(denom);
        Muls(f,f,inv,cols);PipeBarrier<PIPE_V>();Cast(lo,f,RoundMode::CAST_RINT,cols);PipeBarrier<PIPE_V>();Cast(f,lo,RoundMode::CAST_NONE,cols);PipeBarrier<PIPE_V>();
        Mul(h,f,g,cols);PipeBarrier<PIPE_V>();Cast(lo,h,RoundMode::CAST_RINT,cols);PipeBarrier<PIPE_V>();Cast(h,lo,RoundMode::CAST_NONE,cols);PipeBarrier<PIPE_V>();Add(h,h,f,cols);PipeBarrier<PIPE_V>();Cast(lo,h,RoundMode::CAST_RINT,cols);PipeBarrier<PIPE_V>();Cast(h,lo,RoundMode::CAST_NONE,cols);
        DataCopy(ls,gs[cols],cols);PipeBarrier<PIPE_ALL>();Cast(g,ls,RoundMode::CAST_NONE,cols);PipeBarrier<PIPE_V>();Add(f,h,g,cols);PipeBarrier<PIPE_V>();Cast(lo,f,RoundMode::CAST_RINT,cols);PipeBarrier<PIPE_ALL>();DataCopy(go[row*cols],lo,cols);PipeBarrier<PIPE_ALL>();
    }
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" int launch_adarms(void* stream,void* x,void* style,void* out,uint32_t rows,uint32_t cols,float eps) {fused_adarms<<<24,nullptr,stream>>>(x,style,out,rows,cols,eps);return 0;}
#endif

#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void geglu_small_output(GM_ADDR x,GM_ADDR out,uint32_t rows,uint32_t cols) {
    constexpr uint32_t tile=4096;
    TPipe pipe;TBuf<TPosition::VECCALC> ba,bb,bo,bf,bg,bh;
    pipe.InitBuffer(ba,tile*2);pipe.InitBuffer(bb,tile*2);pipe.InitBuffer(bo,tile*2);
    pipe.InitBuffer(bf,tile*4);pipe.InitBuffer(bg,tile*4);pipe.InitBuffer(bh,tile*4);
    auto a=ba.Get<bfloat16_t>();auto b=bb.Get<bfloat16_t>();auto o=bo.Get<bfloat16_t>();
    auto f=bf.Get<float>();auto g=bg.Get<float>();auto h=bh.Get<float>();
    GlobalTensor<bfloat16_t> gx,go;gx.SetGlobalBuffer((__gm__ bfloat16_t*)x);go.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    uint32_t chunks=cols/tile;
    for(uint32_t chunk=GetBlockIdx();chunk<rows*chunks;chunk+=GetBlockNum()) {
        uint32_t row=chunk/chunks,col=(chunk%chunks)*tile;
        DataCopy(a,gx[row*2*cols+col],tile);DataCopy(b,gx[row*2*cols+cols+col],tile);PipeBarrier<PIPE_ALL>();
        Cast(f,a,RoundMode::CAST_NONE,tile);Cast(g,b,RoundMode::CAST_NONE,tile);PipeBarrier<PIPE_V>();
        Mul(h,g,g,tile);PipeBarrier<PIPE_V>();Mul(h,h,g,tile);PipeBarrier<PIPE_V>();Muls(h,h,0.044715f,tile);PipeBarrier<PIPE_V>();Add(h,h,g,tile);PipeBarrier<PIPE_V>();Muls(h,h,-1.5957691216057308f,tile);PipeBarrier<PIPE_V>();Exp(h,h,tile);PipeBarrier<PIPE_V>();Adds(h,h,1.0f,tile);PipeBarrier<PIPE_V>();Div(g,g,h,tile);PipeBarrier<PIPE_V>();Mul(f,f,g,tile);PipeBarrier<PIPE_V>();Cast(o,f,RoundMode::CAST_RINT,tile);PipeBarrier<PIPE_ALL>();DataCopy(go[row*cols+col],o,tile);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" __global__ __aicore__ void geglu_single_output(GM_ADDR x,GM_ADDR out,uint32_t rows,uint32_t cols) {
    constexpr uint32_t tile=8192;
    TPipe pipe;TBuf<TPosition::VECCALC> ba,bb,bo,bf,bg,bh;
    pipe.InitBuffer(ba,tile*2);pipe.InitBuffer(bb,tile*2);pipe.InitBuffer(bo,tile*2);
    pipe.InitBuffer(bf,tile*4);pipe.InitBuffer(bg,tile*4);pipe.InitBuffer(bh,tile*4);
    auto a=ba.Get<bfloat16_t>();auto b=bb.Get<bfloat16_t>();auto o=bo.Get<bfloat16_t>();
    auto f=bf.Get<float>();auto g=bg.Get<float>();auto h=bh.Get<float>();
    GlobalTensor<bfloat16_t> gx,go;gx.SetGlobalBuffer((__gm__ bfloat16_t*)x);go.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    uint32_t chunks=cols/tile;
    for(uint32_t chunk=GetBlockIdx();chunk<rows*chunks;chunk+=GetBlockNum()) {
        uint32_t row=chunk/chunks,col=(chunk%chunks)*tile;
        DataCopy(a,gx[row*2*cols+col],tile);DataCopy(b,gx[row*2*cols+cols+col],tile);PipeBarrier<PIPE_ALL>();
        Cast(f,a,RoundMode::CAST_NONE,tile);Cast(g,b,RoundMode::CAST_NONE,tile);PipeBarrier<PIPE_V>();
        Mul(h,g,g,tile);PipeBarrier<PIPE_V>();Mul(h,h,g,tile);PipeBarrier<PIPE_V>();Muls(h,h,0.044715f,tile);PipeBarrier<PIPE_V>();Add(h,h,g,tile);PipeBarrier<PIPE_V>();Muls(h,h,-1.5957691216057308f,tile);PipeBarrier<PIPE_V>();Exp(h,h,tile);PipeBarrier<PIPE_V>();Adds(h,h,1.0f,tile);PipeBarrier<PIPE_V>();Div(g,g,h,tile);PipeBarrier<PIPE_V>();Mul(f,f,g,tile);PipeBarrier<PIPE_V>();Cast(o,f,RoundMode::CAST_RINT,tile);PipeBarrier<PIPE_ALL>();DataCopy(go[row*cols+col],o,tile);PipeBarrier<PIPE_ALL>();
    }
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" int launch_geglu(void* stream,void* x,void* out,uint32_t rows,uint32_t cols,uint32_t blocks) {if(cols==16384) {geglu_single_output<<<blocks,nullptr,stream>>>(x,out,rows,cols);} else {geglu_small_output<<<blocks,nullptr,stream>>>(x,out,rows,cols);} return 0;}
#endif
