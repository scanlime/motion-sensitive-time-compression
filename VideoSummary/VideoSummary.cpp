#include "pch.h"
#include <ctime>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <npp.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

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

cv::String timestamp()
{
	std::time_t t = std::time(0);
	std::tm tm;
	::localtime_s(&tm, &t);
	char temp[128];
	if (!std::strftime(temp, sizeof temp, "%Y%m%d-%H%M%S", &tm)) {
		abort();
	}
	return cv::String(temp);
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cout << "usage: VideoSummary input_file [threshold] [debug]\n";
		return 1;
	}

	cv::String inputFile = argv[1];
	cv::String outputFile = "F:/recording/summary-" + timestamp() + ".avi";
	const double fps = 30.0;
	unsigned fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
	const double threshold = atof(argc >= 3 ? argv[2] : "0.01");
	const bool debug_output = argc >= 4;
	const int flow_downscale_levels = 2;

	std::cout << "threshold = " << threshold << "\n";
	std::cout << "debug = " << debug_output << "\n";

	// Buffer pools are off by default, must be enabled before creating the first stream,
	// they are used inside many OpenCV algorithms; we want very much to avoid memory
	// allocation in the main loop, as it will cause the API to wait for the GPU.
	cv::cuda::setBufferPoolUsage(true);
	cv::cuda::setBufferPoolConfig(-1, 64 * 1024 * 1024, 32);

	// Single global stream
	cv::cuda::Stream stream;

	// Some OpenCV algorithms are based on the NPP library, which uses a global current stream.
	// To avoid synchronization due to a stream save/restore every time OpenCV uses NPP, we
	// need to preset NPP's current stream to our own.
	nppSetStream(cv::cuda::StreamAccessor::getStream(stream));

	cv::Ptr<cv::cudacodec::VideoReader> input = cv::cudacodec::createVideoReader(inputFile);
	cv::cudacodec::FormatInfo format = input->format();
	std::cout << "Input ok!\n" << inputFile << ", " << format.width << " x " << format.height << "\n";

	cv::Size output_size = cv::Size(format.width, format.height);
	if (debug_output) {
		// Vertically tiled output format
		output_size.height *= 2;
	}
	else if (output_size == cv::Size(1920, 1088)) {
		// Workaround for a bug in cudacodec, size rounded up to the next multiple of 16.
		// This fix is just a hack specific to 1080p video.
		output_size.height = 1080;
	}

	// Our real algorithmic stopping condition here is reaching a target amount of motion
	// per frame, but in order to avoid computing optical flow on every frame we can also
	// take an early out if the number of total foreground pixels is too low. This threshold
	// is based on the overall motion threshold and number of pixels
	int fgmask_threshold = int(threshold * format.width * format.height / 5000);
	std::cout << "mask threshold = " << fgmask_threshold << " pixels\n";

	std::cout << "Writing " << outputFile << " as " << output_size.width << " x " << output_size.height << "\n";
	cv::VideoWriter output(outputFile, fourcc, fps, output_size);

	bool eof = false;
	unsigned input_frames = 0;
	unsigned output_frames = 0;
	unsigned accumulated_frames = 0;
	unsigned flow_frames = 0;
	unsigned rgbx_num_accumulated_frames = 0;
	double motion = 0;

	cv::Ptr<cv::cuda::DenseOpticalFlow> flow_algorithm = cv::cuda::DensePyrLKOpticalFlow::create();
	cv::Ptr<cv::cuda::BackgroundSubtractorMOG> bg_algorithm = cv::cuda::createBackgroundSubtractorMOG();
	cv::Ptr<cv::cuda::Filter> bg_erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, cv::Mat::ones(cv::Size(3, 3), CV_8UC1));

	cv::cuda::GpuMat accumulator(format.height, format.width, CV_32SC4);
	cv::cuda::GpuMat next_accumulator(format.height, format.width, CV_32SC4);
	cv::cuda::GpuMat output_frame(output_size.height, output_size.width, CV_8UC4);
	cv::cuda::GpuMat flowvec(format.height, format.width, CV_32FC2);
	cv::cuda::GpuMat flowvec_magsqr(format.height, format.width, CV_32FC1);

	cv::cuda::GpuMat fgmask_count_buffer(1, 1, CV_32SC1);
	cv::cuda::Event fgmask_count_event;
	int fgmask_count;

	cv::cuda::GpuMat flowvec_sqrsum_buffer(1, 1, CV_64FC1);
	cv::cuda::Event flowvec_sqrsum_event;
	double flowvec_sqrsum;

	cv::cuda::GpuMat flowvec_debug_norm, flowvec_debug_rgbx, fgmask_debug_rgbx;
	cv::cuda::GpuMat rgbx, gray, reference_gray;
	cv::cuda::GpuMat frame, wide_frame;
	cv::cuda::GpuMat fgmask, fgmask_eroded;
	cv::cuda::GpuMat gray_pyramid[flow_downscale_levels];
	cv::cuda::GpuMat fgmask_pyramid[flow_downscale_levels];

	cv::Mat output_buffers[2] = { 
		cv::Mat(format.height, format.width, CV_8UC4),
		cv::Mat(format.height, format.width, CV_8UC4)
	};
	cv::cuda::Event output_events[2];

	// Looping over output frames
	while (!eof) {

		// Looping over input frames
		while (!eof) {
			if (!input->nextFrame(frame)) {
				// No more frames in input, but let the current accumulated output frame finish
				eof = true;
			}
			else {
				input_frames++;
			}

			if (!frame.empty()) {
				// Update a background subtractor model with the original-rate video
				bg_algorithm->apply(frame, fgmask, -1, stream);

				// Eroded foreground mask to look for masses of pixels rather than speckles
				bg_erode->apply(fgmask, fgmask_eroded, stream);

				// Asynchronously start preparing a count of how many foreground pixels are set after eroding.
				// This will be used later as an early out to skip running optical flow on frames that won't have
				// enough motion but we also don't want this operation to stall the pipeline entirely
				cv::cuda::countNonZero(fgmask_eroded, fgmask_count_buffer, stream);
				fgmask_count_buffer.download(cv::Mat(1, 1, CV_32SC1, &fgmask_count), stream);
				fgmask_count_event.record(stream);

				// While that's happening in the background, start preparing inputs to the optical flow.
				// The algorithm doesn't use color information anyway, so we need to convert to gray, and for
				// performance we'll start with a lower resolution image. The eroded mask needs to be scaled
				// by the same amount.
				cv::cuda::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY, 0, stream);
				for (int n = 0; n < flow_downscale_levels; n++) {
					const int level = flow_downscale_levels - 1 - n;
					const int prev = level + 1;
					cv::cudev::pyrDown(n ? gray_pyramid[prev] : gray, gray_pyramid[level], stream);
					cv::cudev::pyrDown(n ? fgmask_pyramid[prev] : fgmask_eroded, fgmask_pyramid[level], stream);
				}

				// Now wait for the fgmask count hopefully while the above conversions and pyrDown are still taking place
				fgmask_count_event.waitForCompletion();

				if (fgmask_count < fgmask_threshold) {
					// Too few pixels set in eroded foreground mask, skip the optical flow entirely
					motion = 0;
				}
				else if (reference_gray.empty()) {
					// No reference yet. Assume no motion for now, set a reference
					gray_pyramid[0].copyTo(reference_gray, stream);
					motion = 0;
				}
				else {
					// Calculate optical flow on each grayscale frame pair
					flow_algorithm->calc(reference_gray, gray_pyramid[0], flowvec, stream);
					cv::cuda::magnitudeSqr(flowvec, flowvec_magsqr, stream);
					flow_frames++;

					// Motion total is average of values squared; calculate it asynchronously
					cv::cuda::calcSum(flowvec_magsqr, flowvec_sqrsum_buffer, fgmask_pyramid[0], stream);					
					flowvec_sqrsum_buffer.download(cv::Mat(1, 1, CV_64FC1, &flowvec_sqrsum), stream);
					flowvec_sqrsum_event.record(stream);

					// Fix me: is there some GPU work we can move here to avoid a stall?

					flowvec_sqrsum_event.waitForCompletion();
					motion = flowvec_sqrsum / (double)flowvec_magsqr.size().area();
				}

				if (motion < threshold) {
					// Not enough motion yet, add this frame to the accumulator
					frame.convertTo(wide_frame, CV_32SC4, stream);
					cv::cuda::add(accumulator, wide_frame, next_accumulator, cv::noArray(), CV_32SC4, stream);
					accumulator.swap(next_accumulator);
					accumulated_frames++;
				}
				else {
					// This frame exceeds the motion threshold. It will be the new reference frame, and we
					// will restart the accumulator with this as the first new frame. The old accumulator
					// is averaged and moved to rgbx.

					if (accumulated_frames >= 1) {
						accumulator.convertTo(rgbx, CV_8UC4, 1.0 / accumulated_frames, 0.0, stream);
						rgbx_num_accumulated_frames = accumulated_frames;
					}

					frame.convertTo(accumulator, CV_32SC4, stream);
					accumulated_frames = 1;

					// Update motion reference frame
					gray_pyramid[0].copyTo(reference_gray, stream);

					// Output a frame
					break;
				}
			}
		}

		if (!rgbx.empty()) {

			if (debug_output) {
				// Visualization of the current foreground mask
				cv::cuda::cvtColor(fgmask_pyramid[0], fgmask_debug_rgbx, cv::COLOR_GRAY2BGRA, 4, stream);

				// Visualize flow magnitude with and without foreground mask
				cv::cuda::normalize(flowvec_magsqr, flowvec_debug_norm, 0, 3 * 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream);
				cv::cuda::cvtColor(flowvec_debug_norm, flowvec_debug_rgbx, cv::COLOR_GRAY2BGRA, 4, stream);

				// Paste together output, starting with an empty frame
				output_frame.setTo(cv::Scalar(0.0), stream);
				cv::Size upper_size = frame.size();
				cv::Size lower_size = flowvec.size();
				cv::cuda::GpuMat upper_frame = output_frame(cv::Rect(cv::Point(0, 0), upper_size));
				cv::cuda::GpuMat lower_frame_0 = output_frame(cv::Rect(cv::Point(0, upper_size.height), lower_size));
				cv::cuda::GpuMat lower_frame_1 = output_frame(cv::Rect(cv::Point(0, upper_size.height + lower_size.height), lower_size));

				// Original resoution RGBX on top
				rgbx.copyTo(upper_frame, stream);

				// Low res debug frames on bottom
				fgmask_debug_rgbx.copyTo(lower_frame_0, stream);
				flowvec_debug_rgbx.copyTo(lower_frame_1, stream);
			}
			else {
				// Non-debug path, but may be cropping the frame
				output_frame = rgbx(cv::Rect(cv::Point(0, 0), output_size));
			}

			// Async download
			unsigned this_buffer = output_frames % 2;
			output_frame.download(output_buffers[this_buffer], stream);
			output_events[this_buffer].record(stream);

			if (output_frames > 0) {
				// Complete the previous frame on the CPU side, hopefully without blocking
				output_events[!this_buffer].waitForCompletion();
				cv::Mat rgb, &rgbx = output_buffers[!this_buffer];
				cv::cvtColor(rgbx, rgb, cv::COLOR_BGRA2BGR);
				output.write(rgb);
			}
			output_frames++;

			std::cout
				<< " " << output_frames
				<< "/" << input_frames
				<< " f" << flow_frames
				<< " x" << rgbx_num_accumulated_frames
				<< " ~" << motion << "\n";
		}
	}

	std::cout << "done.\n";
}