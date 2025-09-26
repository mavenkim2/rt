#include "../common.hlsli"
Texture2D<float> mipInSRV : register(t0);
RWTexture2D<float> mipOutUAV : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 dim;
    mipInSRV.GetDimensions(dim.x, dim.y);
    if (any(DTid.xy * 2 >= dim))
       return;

    float4 rect;
    rect.xy = float2(DTid.xy) * 2;// / dim;
    rect.zw = rect.xy + 1;
    rect /= dim.xyxy;

    float depth00 = mipInSRV.SampleLevel(samplerNearestClamp, rect.xy, 0).r;
    float depth01 = mipInSRV.SampleLevel(samplerNearestClamp, rect.zy, 0).r;
    float depth10 = mipInSRV.SampleLevel(samplerNearestClamp, rect.zw, 0).r;
    float depth11 = mipInSRV.SampleLevel(samplerNearestClamp, rect.xw, 0).r;

    mipOutUAV[DTid.xy] = min(min(depth00, depth01), min(depth10, depth11));
}

