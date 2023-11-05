all:
	g++ main.cpp  Model/Dataloader.cc Model/MatrixMLP/MLP.cc Model/TrainingGround.cc Utility/ActivationFunction.cc -std=c++17 -O3 -o aboba
