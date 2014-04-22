
#include "simpleNN.h"

int main()
{	
	NNParams params;
	params.alpha = 1.5;
	params.lambda = 0.5;

	Simple3LayerNN mynn(400, 25, 10);
	mynn.setParams(params);

	mynn.loadTrainData("data//data.mat");
	mynn.displayData(2.0);	

	mynn.train();

	mynn.displayWeight(4.0);

	/*mynn.loadWeights("data//weights.mat");
	mynn.displayWeight(4.0);*/

	return 1;
}