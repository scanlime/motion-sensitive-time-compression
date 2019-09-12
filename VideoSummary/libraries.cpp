#include "pch.h"

// Direct access to NPP core, for nppSetStream() optimization below
#pragma comment(lib, "nppc")

// OpenCV in debug or release flavor
#ifdef NDEBUG
#pragma comment(lib, "opencv_core412")
#pragma comment(lib, "opencv_cudabgsegm412")
#pragma comment(lib, "opencv_cudaarithm412")
#pragma comment(lib, "opencv_cudafilters412")
#pragma comment(lib, "opencv_cudaimgproc412")
#pragma comment(lib, "opencv_cudaoptflow412")
#pragma comment(lib, "opencv_cudacodec412")
#pragma comment(lib, "opencv_cudawarping412")
#pragma comment(lib, "opencv_videoio412")
#pragma comment(lib, "opencv_imgproc412")
#else
#pragma comment(lib, "opencv_core412d")
#pragma comment(lib, "opencv_cudabgsegm412d")
#pragma comment(lib, "opencv_cudaarithm412d")
#pragma comment(lib, "opencv_cudafilters412d")
#pragma comment(lib, "opencv_cudaimgproc412d")
#pragma comment(lib, "opencv_cudaoptflow412d")
#pragma comment(lib, "opencv_cudacodec412d")
#pragma comment(lib, "opencv_cudawarping412d")
#pragma comment(lib, "opencv_videoio412d")
#pragma comment(lib, "opencv_imgproc412d")
#endif
