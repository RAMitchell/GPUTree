#pragma once
#include <string>
#include <map>
#include <vector>
#include <numeric>

class data{
public:
	std::map<int, std::string> attribute_names;
	std::map<int, std::string> class_names;
	std::vector<std::vector<double>> attributes;
	int class_index = 0;


	void loadARFF(std::string name, int maxItems);


	data(std::string arff, int maxItems = 0){
		loadARFF(arff, maxItems);
		attribute_names[-1] = "--";

	}
};
