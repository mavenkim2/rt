# rt

A physically based monte carlo path tracer used to render [Disney's Moana Island scene](https://disneyanimation.com/resources/moana-island-scene/)
Supports both CPU and interactive NVIDIA GPU rendering.

Currently for personal use only.

## Renders: 
### GPU
Rendered on an RTX 4070 Laptop GPU with 8 GB VRAM. https://www.youtube.com/watch?v=g6yUsX6zboI
### CPU
Rendered in 153 seconds, 1920x804, 64 spp on [Intel 24-core](https://www.lenovo.com/us/en/p/laptops/legion-laptops/legion-5-series/legion-pro-5i-gen-9-16-inch-intel/83df00apus)
![Moana](images/image.bmp)

## GPU Features: 
- Geometry compression using [DGF](https://gpuopen.com/download/DGF.pdf)
- Leverages [RTX Mega Geometry](https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/) to reduce acceleration structure footprint
- [Texture streaming](https://www.yiningkarlli.com/projects/gpuptex/siggraph2025_gpuptex.pdf)

## CPU Features: 
- [Solid angle sampling of spherical rectangle area lights](https://blogs.autodesk.com/media-and-entertainment/wp-content/uploads/sites/162/egsr2013_spherical_rectangle.pdf)
- Multi-level instancing
- N-wide BVH with quantized bounding boxes, compressed pointer info, and SIMD intersection tests
- Adaptive tessellation and displacement of Catmull-Clark subdivision surfaces
- Hit sorting for more coherent texture accesses
- Ray differentials
- Spectral rendering support
- Custom job stealing system, loosely based on [Taskflow](https://github.com/taskflow/taskflow)

## Third-party libraries used: 
- OpenUSD
- Ptex
- STB

## Sources: 
Includes papers, blogs, books, and other links that were helpful.
### Foundational
- [PBRT](https://pbr-book.org/4ed/contents)
- [Robust Monte Carlo Methods for Light Transport Simulation](https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf)

### BVH
- [Compressed-Leaf Bounding Volume Hierarchies](https://www.embree.org/papers/2018-HPG-compressedleafbvh.pdf)
- [Spatial Splits in Bounding Volume Hierarchies](https://www.nvidia.in/docs/IO/77714/sbvh.pdf)


### Light sampling 
- [Importance Sampling of Many Lights with Adaptive Tree Splitting](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf)
- [Cache Points for Production-Scale Occlusion-Aware Many-Lights Sampling and Volumetric Scattering](https://www.yiningkarlli.com/projects/cachepoints/cachepoints.pdf)

### Subdivision, tessellation & displacement
- [Watertight Tessellation using Forward Differencing](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/moreton01_tessellation.pdf)
- [Feature Adadptive GPU Rendering of Catmull-Clark Subdivision Surfaces](https://niessnerlab.org/papers/2012/3feature/niessner2012feature.pdf)
- [Efficient Ray Tracing of Subdivision Surfaces using Tessellation Caching](https://niessnerlab.org/papers/2015/7raytracing/benthin2015efficient.pdf)

### Rendering the Moana Island Scene 
- [https://ingowald.blog/2020/01/09/digesting-the-elephant/](https://ingowald.blog/2020/01/09/digesting-the-elephant/)
- [https://pharr.org/matt/blog/2018/07/08/moana-island-pbrt-1](https://pharr.org/matt/blog/2018/07/08/moana-island-pbrt-1)

### Misc
- [ReSTIR GI](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)
