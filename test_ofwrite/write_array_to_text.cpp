#include "write_array_to_text.h";

int main(int argc, char ** argv){
	int size = 4;
	int subsize = 2;
    // create a vector
    std::vector<std::vector<int>> vec = {};

    for (int i=0; i<size; i++){
		std::vector<int> line = {};
		line.push_back(i);
		for (int j=0; j<subsize; j++)
			line.push_back(-j);
		vec.push_back(line);
    }
	
	std::ofstream sorted_pc_file;
	sorted_pc_file.open("out.txt");
	
	for (auto &x : vec) {
		for (auto &k : x)
			sorted_pc_file << k << " ";
		sorted_pc_file << std::endl;
	}
	sorted_pc_file.close();
    
    return 0;
}