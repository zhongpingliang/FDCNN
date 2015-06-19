#include "sentence_classification.h"
#include "sentence_regression.h"
#include <iostream>
using namespace std;

void test_train();
void test_predict();
int main(int argc, char ** argv){
	SentenceClassification sc;

	sc.train(argc, argv);
	//sc.predict(argc, argv);
	//SentenceRegression sr;
	//sr.train(argc, argv);
	//sr.predict(argc, argv);
}
