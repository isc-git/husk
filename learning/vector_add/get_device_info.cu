#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    int device = 0;

    cudaGetDeviceProperties(&prop, device);

    // dump all the info you could ever need
    printf("name: %s\n", prop.name);
    printf("totalGlobalMem: %ld\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock: %ld\n", prop.sharedMemPerBlock);
    printf("regsPerBlock: %d\n", prop.regsPerBlock);
    printf("warpSize: %d\n", prop.warpSize);
    printf("memPitch: %ld\n", prop.memPitch);
    printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim: {%d, %d, %d}\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize: {%d, %d, %d}\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("clockRate: %d\n", prop.clockRate);
    printf("totalConstMem: %ld\n", prop.totalConstMem);
    printf("major: %d\n", prop.major);
    printf("minor: %d\n", prop.minor);
    printf("textureAlignment: %ld\n", prop.textureAlignment);
    printf("texturePitchAlignment: %ld\n", prop.texturePitchAlignment);
    printf("deviceOverlap: %d\n", prop.deviceOverlap);
    printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf("kernelExecTimeoutEnabled: %d\n", prop.kernelExecTimeoutEnabled);
    printf("integrated: %d\n", prop.integrated);
    printf("canMapHostMemory: %d\n", prop.canMapHostMemory);
    printf("computeMode: %d\n", prop.computeMode);
    printf("maxTexture1D: %d\n", prop.maxTexture1D);
    printf("maxTexture1DMipmap: %d\n", prop.maxTexture1DMipmap);
    printf("maxTexture1DLinear: %d\n", prop.maxTexture1DLinear);
    printf("maxTexture2D: {%d, %d}\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("maxTexture2DMipmap: {%d, %d}\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
    printf("maxTexture2DLinear: {%d, %d, %d}\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);
    printf("maxTexture2DGather: {%d, %d}\n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);
    printf("maxTexture3D: {%d, %d, %d}\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("maxTexture3DAlt: {%d, %d, %d}\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]);
    printf("maxTextureCubemap: %d\n", prop.maxTextureCubemap);
    printf("maxTexture1DLayered: {%d, %d}\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
    printf("maxTexture2DLayered: {%d, %d, %d}\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
    printf("maxTextureCubemapLayered: {%d, %d}\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);
    printf("maxSurface1D: %d\n", prop.maxSurface1D);
    printf("maxSurface2D: {%d, %d}\n", prop.maxSurface2D[0], prop.maxSurface2D[1]);
    printf("maxSurface3D: {%d, %d, %d}\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);
    printf("maxSurface1DLayered: {%d, %d}\n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);
    printf("maxSurface2DLayered: {%d, %d, %d}\n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);
    printf("maxSurfaceCubemap: %d\n", prop.maxSurfaceCubemap);
    printf("maxSurfaceCubemapLayered: {%d, %d}\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);
    printf("surfaceAlignment: %ld\n", prop.surfaceAlignment);
    printf("concurrentKernels: %d\n", prop.concurrentKernels);
    printf("ECCEnabled: %d\n", prop.ECCEnabled);
    printf("pciBusID: %d\n", prop.pciBusID);
    printf("pciDeviceID: %d\n", prop.pciDeviceID);
    printf("pciDomainID: %d\n", prop.pciDomainID);
    printf("tccDriver: %d\n", prop.tccDriver);
    printf("asyncEngineCount: %d\n", prop.asyncEngineCount);
    printf("unifiedAddressing: %d\n", prop.unifiedAddressing);
    printf("memoryClockRate: %d\n", prop.memoryClockRate);
    printf("memoryBusWidth: %d\n", prop.memoryBusWidth);
    printf("l2CacheSize: %d\n", prop.l2CacheSize);
    printf("persistingL2CacheMaxSize: %d\n", prop.persistingL2CacheMaxSize);
    printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("streamPrioritiesSupported: %d\n", prop.streamPrioritiesSupported);
    printf("globalL1CacheSupported: %d\n", prop.globalL1CacheSupported);
    printf("localL1CacheSupported: %d\n", prop.localL1CacheSupported);
    printf("sharedMemPerMultiprocessor: %ld\n", prop.sharedMemPerMultiprocessor);
    printf("regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("managedMemory: %d\n", prop.managedMemory);
    printf("isMultiGpuBoard: %d\n", prop.isMultiGpuBoard);
    printf("multiGpuBoardGroupID: %d\n", prop.multiGpuBoardGroupID);
    printf("singleToDoublePrecisionPerfRatio: %d\n", prop.singleToDoublePrecisionPerfRatio);
    printf("pageableMemoryAccess: %d\n", prop.pageableMemoryAccess);
    printf("concurrentManagedAccess: %d\n", prop.concurrentManagedAccess);
    printf("computePreemptionSupported: %d\n", prop.computePreemptionSupported);
    printf("canUseHostPointerForRegisteredMem: %d\n", prop.canUseHostPointerForRegisteredMem);
    printf("cooperativeLaunch: %d\n", prop.cooperativeLaunch);
    printf("cooperativeMultiDeviceLaunch: %d\n", prop.cooperativeMultiDeviceLaunch);
    printf("pageableMemoryAccessUsesHostPageTables: %d\n", prop.pageableMemoryAccessUsesHostPageTables);
    printf("directManagedMemAccessFromHost: %d\n", prop.directManagedMemAccessFromHost);
    printf("accessPolicyMaxWindowSize: %d\n", prop.accessPolicyMaxWindowSize);
    printf("----- DUMP END -----\n\n");

    printf("Selecting numbers of boxes and threads:\n");
    printf("the device has %d processors\n", prop.multiProcessorCount);
    printf("we can have %d blocks per processor\n", prop.maxBlocksPerMultiProcessor);
    printf("we can have up to %d threads per block, to a max of %d per multiprocessor \n", prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor);

    exit(EXIT_SUCCESS);
}
