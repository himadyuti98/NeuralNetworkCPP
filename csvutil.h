#include <vector>
#include <string>
#include <sstream>
using namespace std;

vector<double> parseCSVLine(string line){
   char delim = ',';
   vector<double> vec;
   string temp;
   stringstream str_strm(line);
   while (std::getline(str_strm, temp, delim)) {
      if(temp.empty())
        continue;
      vec.push_back((double)stoi(temp));
   }
   return vec;
}