#include "data.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>


std::string &strip(std::string &s, const char c) {

    s.erase(std::remove(s.begin(), s.end(), c), s.end());
    return s;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void  data::loadARFF(std::string name, int maxItems) {
    int itemsCount = 0;
    std::string line;
    std::ifstream f(name);

    if (f.is_open()) {

        while (getline(f, line)) {
            if (line.size() == 0) {
                continue;
            }
            else if (line.at(0) == '%') {
                continue;
            }
            else if (line.find("@attribute") != std::string::npos) {


                std::string att_name = split(split(line, ' ')[1], '\'')[0];
                if (att_name == "class") {

                    unsigned first = line.find('{');
                    unsigned last = line.find('}');
                    std::string sub = line.substr(first, last - first);
                    sub = strip(strip(sub, ' '), '{');
                    std::vector<std::string> cn = split(sub, ',');

                    for (auto c : cn) {
                        class_names[class_names.size()] = c;
                    }

                    attribute_names[attribute_names.size()] = att_name;
                    attributes.push_back(std::vector<double>());
                    class_index = attributes.size()-1;

                }
                else {
                    attribute_names[attribute_names.size()] = att_name;
                    attributes.push_back(std::vector<double>());
                }

            }
            else if (line.find("@") == std::string::npos) {
                //Line contains data
                std::vector<std::string> d = split(line, ',');
                for (int i = 0; i < attributes.size(); i++) {
                    if (i == class_index) {

                        for (auto &ci : class_names) {
                            if (d[d.size() - 1] == ci.second) {
                                attributes[i].push_back(ci.first);
                            }
                        }
                    }
                    else {
                        attributes[i].push_back(std::atof(d[i].c_str()));
                    }
                }


                itemsCount++;

                if (itemsCount >= maxItems && maxItems > 0) {
                    break;
                }


            }

        }

        f.clear();

    }
    else {
        std::cerr << "Unable to open file " << name << "\n";
    }

}
