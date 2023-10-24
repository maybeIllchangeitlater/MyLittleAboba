#include <iostream>
#include "dataloader.h"
#include "BlackBox.h"

int main() {
    s21::DataLoader d;
    std::vector<unsigned> t{784, 480, 240, 26};
    s21::Net n(t);
    d.FileToData(s21::DataLoader::kTrain);
    d.CreateSample(10); /// for now i want to overtrain
    size_t epochs = 4;
    for(size_t e = 0; e < epochs; ++e){
        for(size_t current_letter = 0; current_letter < 26; ++current_letter){
//            n.FeedNet(d.Sample()[current_letter][0]);
            std::cout << "Letter is :" << static_cast<char>('a' + current_letter) << std::endl;
            std::cout << "Aboba predicted " << n.GetResult(d.Sample()[current_letter][0]) << std::endl;
            n.BackProp(current_letter);
        }
    } ///feeding it the same 26 letters 10 times to see if there is any result
//    for(size_t current_letter = 0; current_letter < 26; ++current_letter){
//        std::cout << "Letter is :" << static_cast<char>('a' + current_letter) << std::endl;
//        std::cout << "Aboba predicted" << n.GetResult(d.Sample()[current_letter][0]) << std::endl;
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
//    }

    return 0;
}
