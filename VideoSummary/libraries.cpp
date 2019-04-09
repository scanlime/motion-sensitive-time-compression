#include "pch.h"

// Direct access to NPP core, for nppSetStream() optimization below
#pragma comment(lib, "nppc")

// OpenCV in debug or release flavor
#ifdef NDEBUG
#pragma comment(lib, "opencv_core410")
#pragma comment(lib, "opencv_cudabgsegm410")
#pragma comment(lib, "opencv_cudaarithm410")
#pragma comment(lib, "opencv_cudafilters410")
#pragma comment(lib, "opencv_cudaimgproc410")
#pragma comment(lib, "opencv_cudaoptflow410")
#pragma comment(lib, "opencv_cudacodec410")
#pragma comment(lib, "opencv_cudawarping410")
#pragma comment(lib, "opencv_videoio410")
#pragma comment(lib, "opencv_imgproc410")
#else
#pragma comment(lib, "opencv_core410d")
#pragma comment(lib, "opencv_cudabgsegm410d")
#pragma comment(lib, "opencv_cudaarithm410d")
#pragma comment(lib, "opencv_cudafilters410d")
#pragma comment(lib, "opencv_cudaimgproc410d")
#pragma comment(lib, "opencv_cudaoptflow410d")
#pragma comment(lib, "opencv_cudacodec410d")
#pragma comment(lib, "opencv_cudawarping410d")
#pragma comment(lib, "opencv_videoio410d")
#pragma comment(lib, "opencv_imgproc410d")
#endif
