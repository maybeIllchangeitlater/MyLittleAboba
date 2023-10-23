#include <iostream>
#include "dataloader.h"
#include "BlackBox.h"

int main() {
    s21::DataLoader d;
//    std::vector<unsigned> t{784, 128, 64, 26};
//    s21::Net n(t);
//    d.FileToData(s21::DataLoader::kTrain);
//    unsigned alphabet = 26, sets_per_letter = 250, tests_per_letter = 5, counter = 0;
//    unsigned runs = alphabet * sets_per_letter;
//    for(unsigned r = 0; r < runs; ++r) {
//        n.FeedNet(d.Data()[counter].back());
//        n.BackProp(counter);
//        d.Data()[counter].pop_back();
//        if(counter == 25) counter = 0;
//        else ++counter;
//    }
//    std::cout << "Lets see what aboba has learned today";
//    d.FileToData(s21::DataLoader::kTest);
//    runs = alphabet * tests_per_letter;
//    counter = 0;
//    for(unsigned r = 0; r < runs; ++r){
//        std::cout << "letter is " << static_cast<char>('a' + counter) << std::endl;
//        n.FeedNet(d.Data().at(counter).back());
//        if(counter == 25) counter = 0;
//        else ++counter;
//        d.Data().at(counter).pop_back();
    }

    return 0;
}
