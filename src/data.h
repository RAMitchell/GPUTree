#pragma once
#include <string>
#include <map>
#include <vector>
#include <numeric>

class data{
public:
	std::map<int, std::string> attribute_names;
	std::map<int, std::string> class_names;
	std::vector<std::vector<float>> attributes;
	std::vector<float >  attributes_compacted;
	std::vector<char > classes;


	void loadARFF(std::string name, int maxItems);


	data(std::string arff, int maxItems = 0){
		loadARFF(arff, maxItems);
		attribute_names[-1] = "--";

		//Load attributes into a single array
		for (auto&a : attributes){
			attributes_compacted.insert(attributes_compacted.end(), a.begin(), a.end());
		}
	}
};