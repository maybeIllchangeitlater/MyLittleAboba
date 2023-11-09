all:
	g++ main.cpp  Model/Dataloader.cc Model/MLPCore.cc Model/MatrixMLP/MatrixMLP.cc Model/GraphMLP/GraphMLP.cc Model/GraphMLP/GLayer.cc Model/TrainingGround.cc Utility/ActivationFunction.cc -std=c++17 -Wall -Werror -Wextra -O3 -o aboba
