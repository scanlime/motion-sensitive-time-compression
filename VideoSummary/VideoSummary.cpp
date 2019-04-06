#include "pch.h"
#include <ctime>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudev/common.hpp>

#pragma comment(lib, "opencv_core410")
#pragma comment(lib, "opencv_cudabgsegm410")
#pragma comment(lib, "opencv_cudaarithm410")
#pragma comment(lib, "opencv_cudafilters410")
#pragma comment(lib, "opencv_cudaimgproc410")
#pragma comment(lib, "opencv_cudaoptflow410")
#pragma comment(lib, "opencv_cudacodec410")
#pragma comment(lib, "opencv_videoio410")
#pragma comment(lib, "opencv_imgproc410")

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
		std::cout << "usage: VideoSummary input_file [threshold]\n";
		return 1;
	}

	cv::String inputFile = argv[1];
	cv::String outputFile = "F:/recording/summary-" + timestamp() + ".avi";
	const double fps = 30.0;
	unsigned fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
	const double threshold = atof(argc >= 3 ? argv[2] : "0");

	const int flow_crop_left = 64, flow_crop_right = 64;
	const int flow_crop_top = 96, flow_crop_bottom = 64;

	std::cout << "threshold = " << threshold << "\n";

	cv::Ptr<cv::cudacodec::VideoReader> input = cv::cudacodec::createVideoReader(inputFile);
	cv::cudacodec::FormatInfo format = input->format();
    std::cout << "Input ok!\n" << inputFile << ", " << format.width << " x " << format.height << "\n"; 

	// Stacked output images
	cv::Size output_size = cv::Size(format.width, format.height * 3);

	std::cout << "Writing " << outputFile << " as " << output_size.width << " x " << output_size.height << "\n";
	cv::VideoWriter output(outputFile, fourcc, fps, output_size);

	bool eof = false;
	unsigned input_frames = 0;
	unsigned output_frames = 0;
	unsigned accumulated_frames = 0;

	cv::cuda::Stream stream;
	cv::Ptr<cv::cuda::DenseOpticalFlow> flow_algorithm = cv::cuda::DensePyrLKOpticalFlow::create();
	cv::Ptr<cv::cuda::BackgroundSubtractorMOG> bg_algorithm = cv::cuda::createBackgroundSubtractorMOG();
	cv::Ptr<cv::cuda::Filter> bg_erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, cv::Mat::ones(cv::Size(3, 3), CV_8UC1));

	cv::cuda::GpuMat accumulator(format.height, format.width, CV_32SC4);
	cv::cuda::GpuMat next_accumulator(format.height, format.width, CV_32SC4);
	cv::cuda::GpuMat output_frame(output_size.height, output_size.width, CV_8UC4);
	cv::cuda::GpuMat flowvec(format.height, format.width, CV_32FC2);
	cv::cuda::GpuMat flowvec_mag(format.height, format.width, CV_32FC1);
	cv::cuda::GpuMat flowvec_masked(format.height, format.width, CV_32FC1);

	cv::cuda::GpuMat flowvec_norm, flowvec_rgbx;
	cv::cuda::GpuMat rgbx, gray_unmasked, gray, reference_gray;
	cv::cuda::GpuMat frame, wide_frame;
	cv::cuda::GpuMat fgmask, fgmask_rgbx, fgmask_eroded;

	cv::Mat output_buffers[2] = { 
		cv::Mat(format.height, format.width, CV_8UC4),
		cv::Mat(format.height, format.width, CV_8UC4)
	};
	cv::cuda::Event output_events[2];

	while (!eof) {
		// Reset accumulation at startup and after each output frame
		double motion = 0;
		accumulator.setTo(0.0, cv::noArray(), stream);
		accumulated_frames = 0;

		while (!eof) {
			if (!input->nextFrame(frame)) {
				// No more frames in input, but let the current accumulated output frame finish
				eof = true;
			}
			else {
				input_frames++;
			}

			if (!frame.empty()) {
				// Accumulate 32-bit-per-channel input frames, for variable rate frame averaging
				frame.convertTo(wide_frame, CV_32SC4, stream);
				cv::cuda::add(accumulator, wide_frame, next_accumulator, cv::noArray(), CV_32SC4, stream);
				accumulator.swap(next_accumulator);
				accumulated_frames++;

				// Update a background subtractor model with the original-rate video
				bg_algorithm->apply(frame, fgmask, -1, stream);

				// Eroded foreground mask to look for masses of pixels rather than speckles
				bg_erode->apply(fgmask, fgmask_eroded, stream);

				// Grayscale image for optical flow
				cv::cuda::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY, 0, stream);

				if (reference_gray.empty()) {
					gray.copyTo(reference_gray, stream);
				}
				else {
					// Calculate optical flow on each grayscale frame pair
					flow_algorithm->calc(reference_gray, gray, flowvec, stream);

					// Masked flow magnitudes
					cv::cuda::magnitude(flowvec, flowvec_mag, stream);
					flowvec_masked.setTo(cv::Scalar(0.0), cv::noArray(), stream);
					flowvec_mag.copyTo(flowvec_masked, fgmask_eroded, stream);

					// Motion region excluding cropped edges
					cv::cuda::GpuMat motion_region = flowvec_masked(cv::Rect(
						flow_crop_left, flow_crop_top,
						flowvec_masked.cols - flow_crop_left - flow_crop_right,
						flowvec_masked.rows - flow_crop_top - flow_crop_bottom));

					// Motion total is average of values squared
					motion = cv::cuda::sqrSum(motion_region).val[0] / (double)motion_region.size().area();
					if (motion >= threshold) {
						break;
					}
				}
			}
		}

		// Update motion reference frame
		gray.copyTo(reference_gray, stream);

		// Scaled RGB image
		accumulator.convertTo(rgbx, CV_8UC4, 1.0 / accumulated_frames, 0.0, stream);

		// Visualization of the current foreground mask
		cv::cuda::cvtColor(fgmask, fgmask_rgbx, cv::COLOR_GRAY2BGRA, 4, stream);

		// Visualize flow magnitude
		cv::cuda::normalize(flowvec_masked, flowvec_norm, 0, 3*255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream);
		cv::cuda::cvtColor(flowvec_norm, flowvec_rgbx, cv::COLOR_GRAY2BGRA, 4, stream);

		// Paste together output frame
		output_frame.setTo(cv::Scalar(0.0), stream);
		if (!rgbx.empty()) rgbx.copyTo(output_frame.rowRange(format.height * 0, format.height * 1), stream);
		if (!fgmask_rgbx.empty()) fgmask_rgbx.copyTo(output_frame.rowRange(format.height * 1, format.height * 2), stream);
		if (!flowvec_rgbx.empty()) flowvec_rgbx.copyTo(output_frame.rowRange(format.height * 2, format.height * 3), stream);

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

		std::cout << " " << output_frames << "/" << input_frames << " x" << accumulated_frames << " ~" << motion << "\n";
	}

	std::cout << "done.\n";
}