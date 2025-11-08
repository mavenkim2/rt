StructuredBuffer<Photon> photons : register(t0);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
}
