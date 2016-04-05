#pragma once
#include <string>
#include <map>
#include <vector>
#include <numeric>

class Data{
public:
	std::map<int, std::string> attribute_names;
	std::map<int, std::string> class_names;
	std::vector<std::vector<float>> attributes;
	std::vector<int> classes;
	int n_positive;
	int n_negative;


	void loadARFF(std::string name, int maxItems);


	Data(std::string arff, int maxItems = 0){
		loadARFF(arff, maxItems);
		n_positive = std::accumulate(classes.begin(), classes.end(), 0);
		n_negative = classes.size() - n_positive;
		attribute_names[-1] = "--";
	}
};
