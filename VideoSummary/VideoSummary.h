#pragma once

#include <vector>
#include <string>

namespace VideoSummary {

	struct Options {
		Options();

		std::vector<std::string> input_files;
		std::string output_file;
		double threshold;
        double output_fps;
		bool debug;
		bool verbose;
	};

	void run(const Options &opt);
};
