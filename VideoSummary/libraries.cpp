#include "pch.h"

// Direct access to NPP core, for nppSetStream() optimization below
#pragma comment(lib, "nppc")

// OpenCV in debug or release flavor
#ifdef NDEBUG
#pragma comment(lib, "opencv_core411")
#pragma comment(lib, "opencv_cudabgsegm411")
#pragma comment(lib, "opencv_cudaarithm411")
#pragma comment(lib, "opencv_cudafilters411")
#pragma comment(lib, "opencv_cudaimgproc411")
#pragma comment(lib, "opencv_cudaoptflow411")
#pragma comment(lib, "opencv_cudacodec411")
#pragma comment(lib, "opencv_cudawarping411")
#pragma comment(lib, "opencv_videoio411")
#pragma comment(lib, "opencv_imgproc411")
#else
#pragma comment(lib, "opencv_core411d")
#pragma comment(lib, "opencv_cudabgsegm411d")
#pragma comment(lib, "opencv_cudaarithm411d")
#pragma comment(lib, "opencv_cudafilters411d")
#pragma comment(lib, "opencv_cudaimgproc411d")
#pragma comment(lib, "opencv_cudaoptflow411d")
#pragma comment(lib, "opencv_cudacodec411d")
#pragma comment(lib, "opencv_cudawarping411d")
#pragma comment(lib, "opencv_videoio411d")
#pragma comment(lib, "opencv_imgproc411d")
#endif
