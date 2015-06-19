/*******************************************************************
*  Copyright(c) 2014
*  All rights reserved.
*
*  File Name: sentence_regression.h
*  Brief    : This file provides a sentence regression
*     toolkit for tasks such as sentiment regression.
*  Current Version: 1.0
*  Author   : Zhongping Liang
*  Date     : 2014-12-12
******************************************************************/

#ifndef SENTENCE_REGRESSION_H
#define SENTENCE_REGRESSION_H
#include "operator.h"
#include "sentence_model_base.h"
#include <string>
#include <map>
#include <mutex>

using std::string;
using std::mutex;
using std::map;

class SentenceRegression : public SentenceModelBase{
public:
	SentenceRegression();
private:
	SentenceRegression(const SentenceRegression & instance);
	SentenceRegression & operator=(const SentenceRegression & instance);
public:
	void train(integer argc, char **argv);
	void predict(integer argc, char **argv);
private:
	/*	Make dictionary : each feature is has it's own buffer in range feature_table[0, nif)
	*	the dictionary of each feature is dicts[i] where i indicates the ith feature,
	*	the dictionary of label is dicts[nif].
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void make_dict();

	/*	Make dictionary : each feature is has it's own buffer in range feature_table[0, nif)
	*	the dictionary of each feature is dicts[i] where i indicates the ith feature,
	*	the dictionary of label is dicts[nif], the word2vec dict is in dicts[0].
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void make_dict_using_word2vec();

	/*	Save model in model_file_name.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void save_model();

	/*	Load model from model_file_name.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void load_model();

	/*	Save model in model_file_name.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void save_model_binary();

	/*	Load model from model_file_name.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void load_model_binary();

	/*	Get a sentence classifier example from fin: sentence is stored in sen, features is stored in 
	*		features, label is stored in label, length of sentence is stored in lst, line number is stored in linu.
	*	Parameters :
	*		fin		: the input file.
	*		sen		: the vector to store sentence.
	*		feature	: the matrix to store features of sentence.
	*		label	: the label of the sentence.
	*		lst		: the length of input sentence.
	*		linu	: the number of convolution result colmuns per convolution layer.
	*	Return :
	*		void.
	*/
	void get_sentence_regression_example(FILE * fin, string * sen, map<string, integer>::iterator **features, real * label, integer &lst, long long &linu);

	/*	Run train thread.
	*	Parameters :
	*		thrd_id : thread id.
	*	Return :
	*		void.
	*/
	void train_sentence_regression_thread(integer thrd_id);

	/*	Run predict thread.
	*	Parameters :
	*		thrd_id : thread id.
	*	Return :
	*		void.
	*/
	void predict_sentence_regression_thread(integer thrd_id);
private:
	integer prec_exp_cnt;   //predict examples counter.                      
	real total_prec_err;	//total perdict error.
};
#endif