#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<uint> globals : register(u0);
RWStructuredBuffer<PTLAS_INDIRECT_COMMAND> ptlasIndirectCommands : register(u1);

[[vk::push_constant]] PtlasPushConstant pc;

[numthreads(1, 1, 1)]
void main(uint dtID : SV_DispatchThreadID)
{
    if (dtID.x != 0) return;

    uint commandCount = 0;
    uint writeCount = globals[GLOBALS_PTLAS_WRITE_COUNT_INDEX];
    uint updateCount = globals[GLOBALS_PTLAS_UPDATE_COUNT_INDEX];

    if (writeCount)
    {
        PTLAS_INDIRECT_COMMAND command;
        command.opType = PTLAS_TYPE_WRITE_INSTANCE;
        command.argCount = writeCount;
        command.startAddress = pc.writeAddress;
        command.strideInBytes = PTLAS_WRITE_INSTANCE_INFO_STRIDE;

        ptlasIndirectCommands[commandCount] = command;
        commandCount++;
    }
    if (updateCount)
    {
        PTLAS_INDIRECT_COMMAND command;
        command.opType = PTLAS_TYPE_UPDATE_INSTANCE;
        command.argCount = updateCount;
        command.startAddress = pc.updateAddress;
        command.strideInBytes = PTLAS_UPDATE_INSTANCE_INFO_STRIDE;

        ptlasIndirectCommands[commandCount] = command;
        commandCount++;
    }

    globals[GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX] = commandCount;
}

