#include "pch.h"
#include "VideoSummary.h"

VideoSummary::Options::Options() :
    width(1920),
    height(1080),
    motion_threshold(0.1),
    area_threshold(0.0001),
    output_fps(30),
    debug(false),
    verbose(true)
{}

namespace VideoSummary {
    static const double FLOW_IMAGE_SCALE = 1 / 3.0;
    static const int DEBUG_HEIGHT_MULTIPLE = 3;
    static const int BUFFER_STACK_SIZE = 16 * 1024 * 1024;
    static const int BUFFER_STACK_COUNT = 6;

    class VideoSummaryImpl {
    public:
        VideoSummaryImpl(const Options& opt);
        void run(cv::cuda::Stream& stream);

    private:
        const Options& opt;
        int input_file_index;
        bool end_of_input;

        unsigned count_input_frames;
        unsigned count_output_frames;
        unsigned count_optical_flow;

        int accumulator_count;
        int last_accumulator_count;
        double last_motion_sum;
        int fgmask_threshold;

        cv::Ptr<cv::cuda::DenseOpticalFlow> flow_algorithm;
        cv::Ptr<cv::cuda::BackgroundSubtractorMOG> bg_algorithm;
        cv::Ptr<cv::cuda::Filter> bg_erode;

        cv::cuda::GpuMat fgmask, fgmask_eroded, fgmask_eroded_scaled;
        cv::cuda::GpuMat fgmask_wide, fgmask_accum, fgmask_accum_next;

        cv::cuda::GpuMat color_wide, color_accum, color_accum_next;

        cv::cuda::GpuMat debug_fgmask_rgb, debug_fgmask_gray;
        cv::cuda::GpuMat debug_flow_rgb, debug_flow_gray;

        cv::cuda::GpuMat fgmask_count_buffer;
        cv::cuda::Event fgmask_count_event;
        int fgmask_count;

        cv::cuda::GpuMat flowvec_sqrsum_buffer;
        cv::cuda::Event flowvec_sqrsum_event;
        double flowvec_sqrsum;

        FILE* input_reader;
        cv::Mat input_frame;
        cv::cuda::GpuMat input_frame_gpu, input_gray;

        cv::cuda::GpuMat flow_input_frame, flow_reference_frame;
        cv::cuda::GpuMat flowvec, flowvec_magsqr, flowvec_masked;

        FILE* output_writer;
        cv::cuda::GpuMat output_frame;
        cv::Mat output_buffers[2];
        cv::cuda::Event output_events[2];

        void inputRead();
        void outputBegin();
        void outputWrite(cv::cuda::Stream& stream);
        void initThreshold();
        void initAlgorithms();
        void calcForeground(cv::cuda::Stream& stream);
        bool canSkipOpticalFlow();
        void calcOpticalFlow(cv::cuda::Stream& stream);
        void finishMotionSum();
        void commitAccumulator();
        void resetAccumulatorToSingleFrame();
        void printCurrentFrameNumbers();
    };
}

VideoSummary::VideoSummaryImpl::VideoSummaryImpl(const Options& opt) :
    opt(opt),
    input_file_index(0),
    end_of_input(false),
    count_input_frames(0),
    count_output_frames(0),
    count_optical_flow(0),
    accumulator_count(0),
    last_accumulator_count(0),
    last_motion_sum(0)
{}

void VideoSummary::run(const VideoSummary::Options& opt)
{
    VideoSummaryImpl impl(opt);

    // Buffer pools are off by default, must be enabled before creating the first stream,
    // they are used inside many OpenCV algorithms; we want very much to avoid memory
    // allocation in the main loop, as it will cause the API to wait for the GPU.
    cv::cuda::setBufferPoolUsage(true);
    cv::cuda::setBufferPoolConfig(-1, BUFFER_STACK_SIZE, BUFFER_STACK_COUNT);

    // Single global stream
    cv::cuda::Stream stream;

    // Some OpenCV algorithms are based on the NPP library, which uses a global current stream.
    // To avoid synchronization due to a stream save/restore every time OpenCV uses NPP, we
    // need to preset NPP's current stream to our own.
    nppSetStream(cv::cuda::StreamAccessor::getStream(stream));

    impl.run(stream);
}

void VideoSummary::VideoSummaryImpl::inputRead()
{
    HANDLE os_input_reader = INVALID_HANDLE_VALUE;

    while (!end_of_input) {
        if (input_file_index >= opt.input_files.size()) {
            end_of_input = true;
            break;
        }

        // Switching files
        if (!input_reader) {
            const std::string& input_file = opt.input_files[input_file_index];
            const std::string cmd = "ffmpeg -hwaccel qsv -nostats -i \"" + input_file
                + "\" -vf scale=" + std::to_string(opt.width) + ":" + std::to_string(opt.height)
                + " -f fifo -fifo_format rawvideo -map 0:v -vcodec rawvideo -pix_fmt rgb24 -";

            std::cout << cmd << std::endl;
            input_reader = _popen(cmd.c_str(), "rb");
            setvbuf(input_reader, 0, _IONBF, 0);
            os_input_reader = INVALID_HANDLE_VALUE;

            cv::Size frame_size(opt.width, opt.height);
            input_frame.create(frame_size, CV_8UC3);
        }

        // Use low level reads in an attempt to reduce overhead due to FILE* buffering in this read
        if (os_input_reader == INVALID_HANDLE_VALUE) {
            int fd = _fileno(input_reader);
            os_input_reader = (HANDLE)_get_osfhandle(fd);
        }

        DWORD numRead = 0;
        DWORD limit = input_frame.dataend - input_frame.datastart;
        DWORD offset = 0;

        while (input_reader && offset < limit) {
            if (ReadFile(os_input_reader, input_frame.data + offset, limit - offset, &numRead, 0)) {
                offset += numRead;
            }
            else {
                fclose(input_reader);
                input_reader = 0;
                input_file_index++;
            }
        }
        if (input_reader) {
            // finished one frame without EOF
            break;
        }
    }

    if (!end_of_input) {
        count_input_frames++;
    }
}

void VideoSummary::VideoSummaryImpl::outputBegin()
{
    assert(!input_frame.empty());
    cv::Size input_size = input_frame.size();
    cv::Size debug_size(input_size.width, input_size.height * DEBUG_HEIGHT_MULTIPLE);
    cv::Size output_size = opt.debug ? debug_size : input_size;

    output_frame.create(output_size, CV_8UC3);

    if (opt.verbose) {
        std::cout << "Writing " << opt.output_file
            << " as " << output_size.width << " x " << output_size.height
            << std::endl;
    }

    const std::string cmd = "ffmpeg -nostats"
        + std::string(" -f rawvideo -vcodec rawvideo -pix_fmt rgb24 -video_size ")
        + std::to_string(output_size.width) + "x" + std::to_string(output_size.height)
        + std::string(" -framerate ") + std::to_string(opt.output_fps)
        + std::string(" -i - -f fifo -pix_fmt yuv420p -g 60 -c:v libx264 ")
        + std::string("-crf 20 -bufsize 1M -maxrate 12M -f mpegts ")
        + "\"" + opt.output_file + "\"";

    std::cout << cmd << std::endl;
    output_writer = _popen(cmd.c_str(), "wb");
    setvbuf(output_writer, 0, _IONBF, 0);
}

void VideoSummary::VideoSummaryImpl::outputWrite(cv::cuda::Stream& stream)
{
    if (accumulator_count < 1) {
        return;
    }
    last_accumulator_count = accumulator_count;

    cv::cuda::GpuMat color_output_ref;
    if (opt.debug) {
        cv::Size input_size = input_frame.size();

        // Paste together a debug image from stacked frames
        cv::cuda::GpuMat debug_frames[DEBUG_HEIGHT_MULTIPLE];
        for (int n = 0; n < DEBUG_HEIGHT_MULTIPLE; n++) {
            debug_frames[n] = output_frame.rowRange(input_size.height * n, input_size.height * (n + 1));
        }

        // Regular color image on top
        color_output_ref = debug_frames[0];

        // Normalize debug data to fit in 8U
        cv::cuda::normalize(fgmask_accum_next, debug_fgmask_gray, 0, 3 * 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream);
        cv::cuda::normalize(flowvec_masked, debug_flow_gray, 0, 3 * 255, cv::NORM_MINMAX, CV_8UC1, cv::noArray(), stream);

        // Grayscale (normalized) to RGB to match output
        cv::cuda::cvtColor(debug_fgmask_gray, debug_fgmask_rgb, cv::COLOR_GRAY2BGR, 4, stream);
        cv::cuda::cvtColor(debug_flow_gray, debug_flow_rgb, cv::COLOR_GRAY2BGR, 4, stream);

        // Low res debug frames on bottom, scaled up to match
        cv::cudev::resize(debug_fgmask_rgb, debug_frames[1], debug_frames[1].size(), 0, 0, cv::INTER_LINEAR, stream);
        cv::cudev::resize(debug_flow_rgb, debug_frames[2], debug_frames[2].size(), 0, 0, cv::INTER_LINEAR, stream);
    }
    else {
        color_output_ref = output_frame;
    }

    // Convert accumulator to 8U
    color_accum.convertTo(color_output_ref, CV_8UC3, 1.0 / accumulator_count, stream);

    // Async download
    unsigned this_buffer = count_output_frames % 2;
    output_frame.download(output_buffers[this_buffer], stream);
    output_events[this_buffer].record(stream);

    if (count_output_frames > 0) {
        // Complete the previous frame on the CPU side, hopefully without blocking
        output_events[!this_buffer].waitForCompletion();
        cv::Mat& buf = output_buffers[!this_buffer];
        fwrite(buf.datastart, buf.dataend - buf.datastart, 1, output_writer);
    }
    count_output_frames++;
}

void VideoSummary::VideoSummaryImpl::initThreshold()
{
    cv::Size frame_size = input_frame.size();

    fgmask_threshold = frame_size.area() * opt.area_threshold;

    if (opt.verbose) {
        std::cout << "motion threshold = " << opt.motion_threshold << std::endl;
        std::cout << "area threshold = " << opt.area_threshold << " = " << fgmask_threshold << " pixels" << std::endl;
    }
}

void VideoSummary::VideoSummaryImpl::initAlgorithms()
{
    flow_algorithm = cv::cuda::DensePyrLKOpticalFlow::create();
    bg_algorithm = cv::cuda::createBackgroundSubtractorMOG();
    bg_erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, cv::Mat::ones(cv::Size(3, 3), CV_8UC1));
}

void VideoSummary::VideoSummaryImpl::calcForeground(cv::cuda::Stream& stream)
{
    input_frame_gpu.upload(input_frame, stream);

    // Update a background subtractor model with the original-rate video
    bg_algorithm->apply(input_frame_gpu, fgmask, -1, stream);

    // Eroded foreground mask to look for masses of pixels rather than speckles
    bg_erode->apply(fgmask, fgmask_eroded, stream);

    // Scale down a version of the eroded foreground mask to match the size of our optical flow source
    cv::cuda::resize(fgmask_eroded, fgmask_eroded_scaled, cv::Size(0, 0), FLOW_IMAGE_SCALE, FLOW_IMAGE_SCALE, cv::INTER_AREA, stream);

    // During optical flow we want to track not just the instantaneous foreground pixels but the foreground
    // pixels detected during the whole frame averaging group. This frame hasn't been committed to the
    // accumulator yet, so we need to consider the sum of the accumulator and this fgmask we just made.
    // Since that's the same calculation we need for the next frame's accumulator, we will calculate
    // that now and swap it in later.

    fgmask_eroded_scaled.convertTo(fgmask_wide, CV_32FC1, stream);
    if (accumulator_count == 0) {
        fgmask_wide.copyTo(fgmask_accum_next, stream);
    }
    else {
        cv::cuda::add(fgmask_accum, fgmask_wide, fgmask_accum_next, cv::noArray(), CV_32FC1, stream);
    }

    // Asynchronously start preparing a count of how many foreground pixels are set in the
    // combined accumulator fgmask. This will be used later as an early out to skip running optical
    // flow on frames that won't have enough motion.
    cv::cuda::countNonZero(fgmask_accum_next, fgmask_count_buffer, stream);

    // Prepare the color accumulator in the same way, while the fgmask calculation is taking place

    input_frame_gpu.convertTo(color_wide, CV_32SC3, stream);

    fgmask_count_buffer.download(cv::Mat(1, 1, CV_32SC1, &fgmask_count), stream);
    fgmask_count_event.record(stream);

    if (accumulator_count == 0) {
        color_wide.copyTo(color_accum_next, stream);
    }
    else {
        cv::cuda::add(color_accum, color_wide, color_accum_next, cv::noArray(), CV_32SC3, stream);
    }
}

bool VideoSummary::VideoSummaryImpl::canSkipOpticalFlow()
{
    fgmask_count_event.waitForCompletion();
    return fgmask_count < fgmask_threshold;
}

void VideoSummary::VideoSummaryImpl::calcOpticalFlow(cv::cuda::Stream& stream)
{
    // Optical flow does not use color information
    cv::cuda::cvtColor(input_frame_gpu, input_gray, cv::COLOR_BGR2GRAY, 0, stream);

    // Run at a reduced resolution to save time and decrease noise
    cv::cuda::resize(input_gray, flow_input_frame, cv::Size(0, 0), FLOW_IMAGE_SCALE, FLOW_IMAGE_SCALE, cv::INTER_AREA, stream);

    if (flow_reference_frame.empty()) {
        // No reference yet. Assume no motion for now, set a trivial reference so this isn't a special case
        flow_input_frame.copyTo(flow_reference_frame, stream);
    }

    flow_algorithm->calc(flow_reference_frame, flow_input_frame, flowvec, stream);
    cv::cuda::magnitudeSqr(flowvec, flowvec_magsqr, stream);
    count_optical_flow++;

    // Apply the combined foreground mask plus fgmask accumulator
    cv::cuda::multiply(flowvec_magsqr, fgmask_accum_next, flowvec_masked, 1.0 / (1 + accumulator_count), -1, stream);

    // Motion total is average of values squared; calculate it asynchronously
    cv::cuda::calcSum(flowvec_masked, flowvec_sqrsum_buffer, cv::noArray(), stream);
    flowvec_sqrsum_buffer.download(cv::Mat(1, 1, CV_64FC1, &flowvec_sqrsum), stream);
    flowvec_sqrsum_event.record(stream);
}

void VideoSummary::VideoSummaryImpl::finishMotionSum()
{
    // Wait for the squared magnitude. Adjust it according to the number of pixels,
    // so that we're measuring average of the square, and also according to the
    // square of the downscale level, since that changes the length of the measured vectors.

    flowvec_sqrsum_event.waitForCompletion();
    double sqrsum = flowvec_sqrsum;
    assert(sqrsum >= 0.0);

    double sqrarea = flowvec_masked.size().area();

    last_motion_sum = sqrsum / sqrarea;
}

void VideoSummary::VideoSummaryImpl::printCurrentFrameNumbers()
{
    std::cout
        << "[" << opt.input_files[input_file_index]
        << "] " << count_output_frames
        << "/" << count_input_frames;
}

void VideoSummary::VideoSummaryImpl::commitAccumulator()
{
    // To commit to including the current frame in the accumulator
    // rather than discarding, we swap the buffers and increment the count.
    accumulator_count++;
    color_accum_next.swap(color_accum);
    fgmask_accum_next.swap(fgmask_accum);
}

void VideoSummary::VideoSummaryImpl::resetAccumulatorToSingleFrame()
{
    accumulator_count = 1;
    fgmask_wide.swap(fgmask_accum);
    color_wide.swap(color_accum);
    flow_input_frame.swap(flow_reference_frame);
}

void VideoSummary::VideoSummaryImpl::run(cv::cuda::Stream& stream)
{
    inputRead();
    outputBegin();
    initThreshold();
    initAlgorithms();

    while (!end_of_input) {

        calcForeground(stream);
        bool skipFlow = canSkipOpticalFlow();

        if (opt.verbose && opt.debug) {
            printCurrentFrameNumbers();
            std::cout
                << " fg=" << fgmask_count
                << " skip=" << skipFlow
                << std::endl;
        }

        if (canSkipOpticalFlow()) {
            // Current frame has too little motion to run optical flow. 
            // Early out. Commit the accumulator and prepare another frame.

            commitAccumulator();
            inputRead();
            continue;
        }

        // Keep the GPU busy by preparing the next frame while we wait on the motion sum.
        // This will overwrite input_frame with the next frame, while the other buffers
        // still refer to the current frame for a while.

        calcOpticalFlow(stream);
        inputRead();
        finishMotionSum();

        if (opt.verbose && opt.debug) {
            printCurrentFrameNumbers();
            std::cout
                << " ~" << last_motion_sum
                << std::endl;
        }

        if (last_motion_sum < opt.motion_threshold) {
            // Not enough motion yet; keep this frame in the accumulator, move on.
            commitAccumulator();
            continue;
        }

        // This frame meets or exceeds the motion threshold.
        // The current accumulator contents become an output frame, 
        // and this frame starts the next accumulator and becomes the next motion reference.

        outputWrite(stream);
        resetAccumulatorToSingleFrame();

        if (opt.verbose) {
            printCurrentFrameNumbers();
            std::cout
                << " f" << count_optical_flow
                << " x" << last_accumulator_count
                << " ~" << last_motion_sum
                << std::endl;
        }
    }
}

