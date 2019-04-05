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

	const int flow_crop_left = 32, flow_crop_right = 32;
	const int flow_crop_top = 96, flow_crop_bottom = 64;

	// Default threshold is zero, which skips no frames (for inspecting motion values).
	// When zero or negative, debug mode is active.
	std::cout << "threshold = " << threshold << "\n";

	cv::Ptr<cv::cudacodec::VideoReader> input = cv::cudacodec::createVideoReader(inputFile);
	cv::cudacodec::FormatInfo format = input->format();
    std::cout << "Input ok!\n" << inputFile << ", " << format.width << " x " << format.height << "\n"; 

	// Workaround what seems to be a cv::cudacodec bug, where the format does not distinguish
	// between frame size and buffer size. I'm usually running this on 1080p video, which h264
	// will store as 1920x1088 internally. We see this size instead of the file's true size
	// of 1920x1080. This fixes the one particular case only.
	cv::Size output_size = cv::Size(format.width, format.height);
	if (output_size == cv::Size(1920, 1088)) {
		output_size.height = 1080;
	}

	std::cout << "Writing " << outputFile << " as " << output_size.width << " x " << output_size.height << "\n";
	cv::VideoWriter output(outputFile, fourcc, fps, output_size);

	bool eof = false;
	unsigned input_frames = 0;
	unsigned output_frames = 0;
	unsigned accumulated_frames = 0;
	unsigned min_accumulated_frames = 1;

	cv::cuda::Stream stream;
	cv::Ptr<cv::cuda::DenseOpticalFlow> flow_algorithm = cv::cuda::FarnebackOpticalFlow::create(3, 0.5, true, 13, 20);

	cv::cuda::GpuMat accumulator(format.height, format.width, CV_32SC4);
	cv::cuda::GpuMat next_accumulator(format.height, format.width, CV_32SC4);
	cv::cuda::GpuMat flowvec, flowvec_mag;
	cv::cuda::GpuMat rgbx, gray_pyr[3], gray_flow, last_gray_flow;
	cv::cuda::GpuMat frame;
	cv::cuda::GpuMat wide_frame;

	cv::Mat sys_rgbx[2] = { 
		cv::Mat(format.height, format.width, CV_8UC4),
		cv::Mat(format.height, format.width, CV_8UC4)
	};
	cv::cuda::Event sys_rgbx_done[2];

	while (!eof) {
		// Reset accumulation at startup and after each output frame
		double motion = 0;
		accumulator.setTo(0.0, cv::noArray(), stream);
		accumulated_frames = 0;

		do {
			if (!input->nextFrame(frame)) {
				// No more frames in input, but let the current accumulated output frame finish
				eof = true;
			} else {
				// Accumulate input frames
				input_frames++;
				frame.convertTo(wide_frame, CV_32SC4, stream);
				cv::cuda::add(accumulator, wide_frame, next_accumulator, cv::noArray(), CV_32SC4, stream);
				accumulator.swap(next_accumulator);
				accumulated_frames++;
			}

			if (accumulated_frames >= min_accumulated_frames) {
				// Limit how quickly the number of frames averaged can decrease.
				// Sometimes we may want to be examining single frames, but often we want
				// to reduce outliers and noise by averaging several frames.
				// We can also go faster if we skip optical flow frames.

				min_accumulated_frames = std::max<unsigned>(1, accumulated_frames * 0.75);

				// Scaled RGB image
				accumulator.convertTo(rgbx, CV_8UC4, 1.0 / accumulated_frames, 0.0, stream);

				// Grayscale version for optical flow
				cv::cuda::cvtColor(rgbx, gray_pyr[0], cv::COLOR_BGRA2GRAY, 0, stream);
				cv::cuda::pyrDown(gray_pyr[0], gray_pyr[1], stream);
				gray_pyr[1].convertTo(gray_flow, CV_32FC1, stream);

				if (last_gray_flow.empty()) {
					// No reference image yet
					motion = fabs(threshold);
					break;
				}
				else {
					// Peak filtered optical flow vs reference
					flow_algorithm->calc(last_gray_flow, gray_flow, flowvec, stream);
					flowvec.adjustROI(-flow_crop_top, -flow_crop_bottom, -flow_crop_left, -flow_crop_right);
					cv::cuda::magnitude(flowvec, flowvec_mag, stream);
					motion = cv::cuda::sqrSum(flowvec_mag).val[0] / (double)flowvec_mag.size().area();
				}
			}

		} while (threshold != 0.0 && motion < fabs(threshold) && !eof);

		// Save reference frame
		gray_flow.copyTo(last_gray_flow, stream);

		// Async download
		unsigned this_buffer = output_frames % 2;
		rgbx.download(sys_rgbx[this_buffer], stream);
		sys_rgbx_done[this_buffer].record(stream);

		if (output_frames > 0) {
			// Complete the previous frame on the CPU side, hopefully without blocking
			sys_rgbx_done[!this_buffer].waitForCompletion();
			cv::Mat &rgbx = sys_rgbx[!this_buffer];

			if (threshold <= 0.0 && !flowvec.empty()) {
				// Debug mode; draw the calculated optical flow
				cv::cuda::GpuMat gpu_flow_gray, gpu_flow_rgbx;
				cv::cuda::normalize(flowvec_mag, gpu_flow_gray, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream);
				cv::cuda::cvtColor(gpu_flow_gray, gpu_flow_rgbx, cv::COLOR_GRAY2BGRA, 4, stream);
				cv::Mat cpu_flow(gpu_flow_rgbx);
				cpu_flow.copyTo(rgbx(cv::Rect(cv::Point(flow_crop_left, flow_crop_top), cpu_flow.size())));
			}

			cv::Mat rgb;
			cv::cvtColor(rgbx, rgb, cv::COLOR_BGRA2BGR);
			rgb.adjustROI(0, output_size.height - rgb.rows, 0, output_size.width - rgb.cols);
			output.write(rgb);
		}
		output_frames++;

		std::cout << " " << output_frames << "/" << input_frames << " x" << accumulated_frames << " ~" << motion << "\n";
	}

	std::cout << "done.\n";
}