/*******************************************************************
*  Copyright(c) 2014
*  All rights reserved.
*
*  File Name: sentence_model_base.h
*  Brief    : This file provides a sentence model basic class.
*  Current Version: 1.0
*  Author   : Zhongping Liang
*  Date     : 2014-12-12
******************************************************************/

#ifndef SENTENCE_MODELP_BASE_H
#define SENTENCE_MODELP_BASE_H
#include "operator.h"
#include <string>
#include <map>
#include <mutex>

using std::string;
using std::mutex;
using std::map;

class SentenceModelBase{
public:
	SentenceModelBase();
protected:
	SentenceModelBase(const SentenceModelBase & instance);
	virtual SentenceModelBase & operator=(const SentenceModelBase & instance);
public:
	virtual void train(integer argc, char **argv) = 0;
	virtual void predict(integer argc, char **argv) = 0;
protected:
	/*	Read configuration from config file.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	void read_config();

	/*	Check configuration.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	void check_config();

	/*	Initialize the networks.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	void init();

	/*	Forword propagate.
	*	Parameters :
	*		SENM	: the matrix of input sentence.
	*		lst		: the length of input sentence.
	*		CM		: the matrix of convolution result .
	*		ncc		: the number of convolution result colmuns per convolution layer.
	*		kpv		: the vector of k-max pooling result.
	*		ndk		: the number of dynamic k-max pooling results per layer
	*		hve		: the vector of hidden layer.
	*		ove		: the vector of out layer.
	*	Return :
	*		void.
	*/
	void forward(real * SENM, integer lst, real***CM, integer * ncc, integer ***kpv, integer *ndk, real ** hve, real *ove);

	/*	Backward propagate.
	*	Parameters :
	*		SENM	: the matrix of input sentence.
	*		lsz		: the matrix of input sentence's error term.
	*		lst		: the length of input sentence.
	*		CM		: the matrix of convolution result.
	*		lcz		: the matrix of convolution result's error term.
	*		ncc		: the number of convolution result colmuns per convolution layer.
	*		kpv		: the vector of k-max pooling result.
	*		ndk		: the number of dynamic k-max pooling results per layer
	*		hve		: the vector of hidden layer.
	*		lhz		: the vector of hidden layer's error term.
	*		loz		: the vector of out layer's error term.
	*	Return :
	*		void.
	*/
	void backward(real * SENM, real *lsz, integer lst, real ***CM, real ***lcz, integer *ncc, integer ***kpv, integer *ndk, real **hve, real**lhz, real * loz, real * lche);

	/*	Make dictionary : each feature is has it's own buffer in range feature_table[0, nif)
	*	the dictionary of each feature is dicts[i] where i indicates the ith feature,
	*	the dictionary of label is dicts[nif].
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void make_dict() = 0;

	/*	Make dictionary : each feature is has it's own buffer in range feature_table[0, nif)
	*	the dictionary of each feature is dicts[i] where i indicates the ith feature,
	*	the dictionary of label is dicts[nif], the word2vec dict is in dicts[0].
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void make_dict_using_word2vec() = 0;

	/*	Calculate pooling result number.
	*	Parameters :
	*		sen_len	: the length of sentence.
	*		cncl	: current convolution layer number
	*	Return :
	*		void.
	*/
	integer dynamic_k(integer sen_len, integer cncl);

	/*	Save model in model_file_name.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void save_model() = 0;

	/*	Load model from model_file_name.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void load_model() = 0;

	/*	Save model in model_file_name.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void save_model_binary() = 0;

	/*	Load model from model_file_name.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	virtual void load_model_binary() = 0;

	/*	Allocate memory for parameters of networks.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	void alloc_mem();

	/*	Release memory for parameters of networks and dictionaries.
	*	Parameters :
	*		void.
	*	Return :
	*		void.
	*/
	void free_men();
	/*	Look up table: search dictionaries and store parameters in SENM.
	*	Parameters :
	*		features: the iteration of features in dictionaries.
	*		SENM	: the matrix of sentence
	*		lst		: the length of sentence
	*	Return :
	*		void.
	*/
	void look_up_table(map<string, integer>::iterator **features, real *SENM, integer lst);

	/*	Update table: update table using error term.
	*	Parameters :
	*		features: the iteration of features in dictionaries.
	*		SENM	: the matrix of sentence
	*		lst		: the length of sentence
	*	Return :
	*		void.
	*/
	void update_feature_table_with_regularization(map<string, integer>::iterator **features, real *lsz, integer lst);

	/*	Update table: update table using error term.
	*	Parameters :
	*		features: the iteration of features in dictionaries.
	*		SENM	: the matrix of sentence
	*		lst		: the length of sentence
	*	Return :
	*		void.
	*/
	void update_feature_table_with_regularization_using(map<string, integer>::iterator **features, real *lsz, integer lst);

	/*	Get argument position from argv.
	*	Parameters :
	*		str		: the string to be seeked.
	*		argc	: the number of argument.
	*		argv	: the string vector of argument
	*	Return :
	*		void.
	*/
	integer get_arg_pos(char *str, integer argc, char ** argv);

	/*	Get a float random value: the return value is in range(-1/layer_size, 1/layer_size).
	*	Parameters :
	*		layer_size	: Limit the range of random value.
	*	Return :
	*		float :a float random value in range(-1/layer_size, 1/layer_size).
	*/
	real get_rand(integer layer_size);

protected:
	integer nwd;			//number of word dimensions
	integer ncl;			//number of convolution layers
	integer *scw;			//size of convolution window  per convolution layer,11 8 5, with a attenuation of Watt
	integer *ncr;			//number of convolution result rows per convolution layer, maybe calucateed by max{Dtop, upper( (ncl - l) * nwd / ncl ) } 
	real ***MM;				//convolution filter matrix per convolution layer, there are ncm matrixs in each convolution layer, so the MM number is ncm[i] * ncm[i-1] , with dimension of -MM[0]:dcr[0] * (scw[0] * nwd + 1) -MM[i]:dcr[i] * (scw[i] * dcr[i-1] + 1) ; note a bias is added to it
	integer *ncm;			//number of CM matrix per convolution layer.
	integer Ktop;			//number of k-max pooling result in the top layer
	integer nhl;			//number of hidden layers
	integer *nhr;			//dimension of hidden result per layer
	real **WM;				//weight matrix per hidden layer WM[i],with dimension of -WM[0] : nhr[0] * (nid + 1) -WM[i] : nhr[i] * (nhr[i-1] + 1); note a bias is added to it
	real *U;				//weight matrix of hidden to output layer, with dimension of U: nor * (nhr[nhl - 1] + 1);  note a bias is added to it
	integer nor;			//number of output layer dimension
	real alp;				//learning rate
	real lmd;				//regularization rate
	integer * nfd;			//number of each feature dimension
	integer nif;			//number of input features
	integer nte;			//number of trainning examples
	real **feature_table;	//featuer table.
	integer iter_num;		//number of iteration times
	integer thrd_num;		//number of threads.
	long long train_line_num;	//number of lines in train file.
	long long exa_num;			//number of examples that have beed trained.
	real avgerr = 0;				//average loss error per LOG_PER_EXA
	map<string, integer> * dicts;	//dicts to store features and labels, the number of dicts is nif + 1, whose the last dict is used to save labels.
	char config_file_name[MAX_WORD_LEN];		//configuration file name.
	char train_file_name[MAX_WORD_LEN];			//train file name.
	char model_file_name[MAX_WORD_LEN];			//model file name.
	char predict_in_file_name[MAX_WORD_LEN];	//prediction in file name.
	char predict_out_file_name[MAX_WORD_LEN];	//prediction out file name.
	char word2vec_file_name[MAX_WORD_LEN];		//prediction out file name.
	FILE * train_fin;		// train file
	FILE * pred_fin;		// predict in file
	FILE * pred_fout;		// predict out file
	FILE * model_file;		// model file
	FILE * config_file;		// config file
	FILE * word2vec_fin;	// word2vec file
	mutex mtx_tfi;			// mutex of train file.
	mutex mtx_pfi;			// mutex of train file.
	mutex mtx_pfo;			// mutex of train file.
	long long linu;			// line number
	integer citer;			// current iteration number
	integer bin;			// binary save or load mode
};
#endif