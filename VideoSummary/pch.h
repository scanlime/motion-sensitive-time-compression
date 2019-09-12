#ifndef PCH_H
#define PCH_H

// Standard
#include <time.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

// NVidia NPP
#include <npp.h>

// Bundled third party libraries
#include "clipp.h"

// OpenCV 4 with CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#endif //PCH_H
