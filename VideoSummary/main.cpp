#include "pch.h"
#include "VideoSummary.h"

using namespace clipp;
using namespace std;

int main(int argc, char **argv)
{
    VideoSummary::Options opts;
    bool allow_overwrite = false;
    const std::string output_extension = "avi";

    auto cli = (
        values("inputs", opts.input_files) % "One or more video files to process sequentially",
        option("-o", "--output") & value("output", opts.output_file) % (
            "Use a specific output file or output directory; "
            "If this is a directory, the name will be chosen automatically. "
            "Defaults to the same directory as the first input file."),
        option("-t", "--threshold") & value("threshold", opts.threshold) % (
            "Change the per-frame motion threshold [default = "
            + to_string(opts.threshold) + "]"),
        option("-r", "--fps") & value("rate", opts.output_fps) % (
            "Set the frame rate used by the output file [default = "
            + to_string(opts.output_fps) + "]"),
        option("-q", "--quiet").set(opts.verbose, false) % "Suppress normal status output",
        option("-d", "--debug").set(opts.debug) % "Generate a larger output frame that visualizes algorithm results",
        option("-y", "--overwrite").set(allow_overwrite) % "Overwrite output file if it exists"
        );

    if (!parse(argc, argv, cli) || opts.input_files.empty()) {
        cout << make_man_page(cli, "VideoSummary");
        return 1;
    }

    // Make sure all inputs are readable before we start
    bool any_inputs_failed = false;
    for_each(opts.input_files.begin(), opts.input_files.end(), [&](const string &filename) {
        ifstream teststream(filename, ifstream::binary);
        if (teststream.fail()) {
            cerr << "ERROR: Input file can't be opened, " << filename << endl;
            any_inputs_failed = true;
        }
    });
    if (any_inputs_failed) {
        return 1;
    }

    // If no output was specified, default to the first input's directory
    if (opts.output_file.empty()) {
        opts.output_file = filesystem::path(opts.input_files[0]).remove_filename().string();
    }

    // If the output path is a directory, add an automatic filename
    if (opts.output_file.empty() || filesystem::is_directory(opts.output_file)) {
        ostringstream name;

        tm tm;
        time_t t = std::time(0);
        localtime_s(&tm, &t);
        name << put_time(&tm, "summary-%Y%m%d%H%M%S-");
   
        filesystem::path input_base = filesystem::path(opts.input_files[0]).filename();
        while (input_base.has_extension()) {
            input_base.replace_extension();
        }
        name << input_base.string();

        name << "-t" << opts.threshold;

        if (opts.debug) {
            name << "-d";
        }

        name << "." << output_extension;

        opts.output_file = (filesystem::path(opts.output_file) / name.str()).make_preferred().string();
    }

    // If the output already exists, it's an error unless we allow overwrite
    if (filesystem::exists(opts.output_file) & !allow_overwrite) {
        cerr << "ERROR: Refusing to overwrite output file without -y option, "
            << opts.output_file << endl;
        return 1;
    }

    try {
        VideoSummary::run(opts);
    } 
    catch (const std::exception &err) {
        cerr << "ERROR: " << err.what() << endl;
        return 1;
    }

    cout << "done." << endl;
    return 0;
}