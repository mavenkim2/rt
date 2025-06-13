struct Queue 
{
};

globallycoherent RWStructuredBuffer Queue queue : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    for (;;)
    {
        // Pop node from queue
    }
}

