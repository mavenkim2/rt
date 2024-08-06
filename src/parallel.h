struct Light;
struct WorkItem
{
    u32 startX;
    u32 startY;
    u32 onePastEndX;
    u32 onePastEndY;
};

struct RenderParams
{
    Primitive bvh;
    Image *image;
    Light *lights;
    vec3 cameraCenter;
    vec3 pixel00;
    vec3 pixelDeltaU;
    vec3 pixelDeltaV;
    vec3 defocusDiskU;
    vec3 defocusDiskV;
    f32 defocusAngle;
    u32 maxDepth;
    u32 samplesPerPixel;
    u32 squareRootSamplesPerPixel;
    u32 numLights;
};

struct WorkQueue
{
    WorkItem *workItems;
    RenderParams *params;
    u64 volatile workItemIndex;
    u64 volatile tilesFinished;
    u32 workItemCount;
};

bool RenderTile(WorkQueue *queue);

THREAD_ENTRY_POINT(WorkerThread)
{
    WorkQueue *queue = (WorkQueue *)parameter;
    while (RenderTile(queue)) continue;
}
