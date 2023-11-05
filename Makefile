all:
	g++ main.cpp  Model/Dataloader.cc Model/MatrixMLP/MatrixMLP.cc Model/TrainingGround.cc Utility/ActivationFunction.cc -std=c++17 -Wall -Werror -Wextra -O3 -o aboba
