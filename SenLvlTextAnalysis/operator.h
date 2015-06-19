/*******************************************************************
*  Copyright(c) 2014
*  All rights reserved.
*
*  File Name: operator.h
*  Brief    : This file provides some basic operating
*     functions for convolution neural network.
*  Current Version: 1.0
*  Author   : Zhongping Liang
*  Date     : 2014-12-12
******************************************************************/

#ifndef OPERATOR_H  
#define OPERATOR_H  

#include <string>
#include <cstdint>
#include <cmath>

using std::string;

typedef double real;
typedef int32_t integer;

const integer MAX_SEN_LEN = 10000;
const integer MAX_WORD_LEN = 100;
const integer MAX_FIELD_NUM = 100;
const integer LOG_PER_EXA = 1000;
const string NIL = "NIL";

/*	Get the minimum integer that is not smaller than x.
*	Parameters :
*		x			: the float value x.
*	Return :
*		integer :the minimum integer that is not smaller than x
*/
inline integer upper(real x){
	return x > static_cast<integer>(x) ? static_cast<integer>(x)+1 : static_cast<integer>(x);
}

/*	Get the square of x.
*	Parameters :
*		x			: x.
*	Return :
*		float :The square of x.
*/
inline real square(real x){
	return x * x;
}

/*	Get the absolute value of x.
*	Parameters :
*		x			: x.
*	Return :
*		float :The absolute value of x.
*/
/*
inline real abs(real x)
{
	return x < 0 ? -x : x;
}
*/

/*	Provide a integer operator for qsort.
*	Parameters :
*		x			: x.
*		y			: y.
*	Return :
*		integer : x - y.
*/
inline integer smaller(const void * x, const void * y){
	return *((integer*)x) - *((integer*)y);
}

/*	Get the larger value of x and y.
*	Parameters :
*		x			: x.
*		y			: y
*	Return :
*		integer :The larger value of x and y.
*/
inline integer max(integer x, integer y) {
	return x < y ? y : x;
}

/*	Provide a non linear function for the networks: here the function is tanh.
*	Parameters :
*		x			: x.
*	Return :
*		integer : tanh(x).
*/
inline real non_linear_function(real x){
	//tanh
	//real a = exp(x);
	//real b = exp(-x);
	//return (a - b) / (a + b);

	return tanh(x);

	/*
	//hard tanh
	if (x > 1) return 1;
	if (x < -1) return -1;
	return x;
	*/
	/*
	//improve tanh
	if (x > 3) return 1;
	if (x < -3) return -1;
	real a = exp(x);
	real b = exp(-x);
	return (a - b) / (a + b);
	*/
}

/*	Calculate the derivative of the non linear function: here the function is tanh.
*	Parameters :
*		x			: x.
*	Return :
*		integer : 1 - x * x.
*/
inline real non_linear_function_derivative(real x){
	/*
	//tanh
	return 1 - square(x);
	*/

	/*
	//hard tanh;
	return 1;
	*/

	//improve tanh
	return 1 - square(x);
}

/*	Calcute convolution : input_M ** weight_M , result is saved in output_M. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		output_M	: the result matrix.
*	Return :
*		void
*/
void convolution(const real * input_M, integer nir, integer nic, const real * weight_M, integer nwr, integer wsz, real * output_M);

/*	Calcute convolution and add a non_linear_function : f(input_M ** weight_M) , result is saved in output_M. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		output_M	: the result matrix.
*	Return :
*		void
*/
void convolution_and_non_linear(const real * input_M, integer nir, integer nic, const real * weight_M, integer nwr, integer wsz, real * output_M);

/*	Calcute convolution with pooling result : input_M ** weight_M , result is saved in output_M. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		output_M	: the result matrix.
*		pci			: pooling columns index vector
*		dpk			£ºnumber of pooling columns
*	Return :
*		void
*/
void convolution_with_pooling_result(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M, integer * pci, integer dpk);

/*	Calcute convolution with pooling result and add a non_linear_function : f(input_M ** weight_M) , result is saved in output_M. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		output_M	: the result matrix.
*		pci			: pooling columns index vector
*		dpk			£ºnumber of pooling columns
*	Return :
*		void
*/
void convolution_with_pooling_result_and_non_linear(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M, integer * pci, integer dpk);

/*	Calcute convolution with pooling result and accumulate to output_M : output_M += input_M ** weight_M . Note a bias is considered in the weight matrix.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		output_M	: the result matrix.
*		pci			: pooling columns index vector
*		dpk			£ºnumber of pooling columns
*	Return :
*		void
*/
void convolution_with_pooling_result_accumulate(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M, integer * pci, integer dpk);

/*	Calcute convolution with pooling result and accumulate to output_M , and add a non_linear_function: output_M = f(output_M += input_M ** weight_M) . Note a bias is considered in the weight matrix.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		output_M	: the result matrix.
*		pci			: pooling columns index vector
*		dpk			£ºnumber of pooling columns
*	Return :
*		void
*/
void convolution_with_pooling_result_accumulate_and_non_linear(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M, integer * pci, integer dpk);

/*	Calcute weighted sum : weight_M * input_V , result is saved in output_V. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		output_V	: the result vector.
*	Return :
*		void
*/
void weighted_sum(const real * input_V, integer nir, const real * weight_M, integer nwr, real * output_V);

/*	Calcute weighted sum and add a non_linear_function : weight_M * input_V , result is saved in output_V. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		output_V	: the result vector.
*	Return :
*		void
*/
void weighted_sum_and_non_linear(const real * input_V, integer nir, const real * weight_M, integer nwr, real * output_V);

/*	Calcute weighted sum with pooling results : weight_M * (input_M, ...) , result is saved in output_V. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		output_V	: the result vector.
*		pci			: pooling columns index vector
*		dpk			£ºnumber of pooling columns
*	Return :
*		void
*/
void weighted_sum_with_pooling_results(real ** input_M, integer nim, integer nir, integer nic, const real * weight_M, integer nwr, real * output_V, integer ** pci, integer dpk);

/*	Calcute weighted sum with pooling result and add a non_linear_function: f(weight_M * (input_M, ...)) , result is saved in output_V. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		nwr			: the number of weight matrix rows.
*		output_V	: the result vector.
*		pci			: pooling columns index vector
*		dpk			£ºnumber of pooling columns
*	Return :
*		void
*/
void weighted_sum_with_pooling_results_and_non_linear(real ** input_M, integer nim, integer nir, integer nic, const real * weight_M, integer nwr, real * output_V, integer ** pci, integer dpk);

/*	Do k max pooling operation : find top k column vectors compared by their lenght |x|, and keep their order.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input vector rows.
*		nic			: the number of input vector columns.
*		output_V	: the result vector to save the index of each top k column vector.
*		k			: the number of top results.
*	Return :
*		void
*/
void k_max_pooling(const real * input_M, integer nir, integer nic, integer * output_V, integer k);

/*	Calcute weighted sum and a derivative item is multiplied : T(weight_M) * input_V , result is saved in output_V. Note a bias is considered in the weight matrix.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		derivative_V: the derivative vector
*		ndr			: the number of derivative vector rows.
*		output_V	: the result vector.
*	Return :
*		void
*/
void weighted_sum_error_term_and_derivative(const real * input_V, integer nir, const real * weight_M, real * derivative_V, integer ndr, real * output_V);

/*	Calcute weighted sum and a derivative item is multiplied, also weights and biases are updated: T(weight_M) * input_V , result is saved in output_V. Note a bias is considered in the weight matrix.
*   This function is used to back propagation the error term from layer l+1 to layer l.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		derivative_V: the derivative vector
*		ndr			: the number of derivative vector rows.
*		output_V	: the result vector.
*		alpha		: the learning rate.
*	Return :
*		void
*/
void weighted_sum_error_term_and_derivative_and_update_weight(const real * input_V, integer nir, real * weight_M, real * derivative_V, integer ndr, real * output_V, real alpha);

/*	Calcute weighted sum and a derivative item is multiplied, also weights and biases are updated: T(weight_M) * input_V , result is saved in output_V. Note a bias is considered in the weight matrix.
*   This function is used to back propagation the error term from layer l+1 to layer l.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		derivative_V: the derivative vector
*		ndr			: the number of derivative vector rows.
*		output_V	: the result vector.
*		alpha		: the learning rate.
*		lambda		: the regularization rate.
*	Return :
*		void
*/
void weighted_sum_error_term_and_derivative_and_update_weight_with_regularization(const real * input_V, integer nir, real * weight_M, real * derivative_V, integer ndr, real * output_V, real alpha, real lambda);

/*	Calcute weighted sum with pooling result and a derivative item is multiplied , also weights and biases are updated: T(weight_M) * input_V , result is saved in output_V. Note a bias is considered in the weight matrix.
*   This function is used to back propagation the error term from layer h[0] to layer Ctop.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		derivative_M: the derivative matrix.
*		ndm			: the number of derivative matrix.
*		ndr			: the number of derivative vector rows.
*		ndc			: the number of derivative vector columns.
*		output_M	: the result vector.
*		pci			: the pooling result vector.
*		dpk			: the number of pooling columns
*		alpha		: the learning rate.
*	Return :
*		void
*/
void weighted_sum_error_term_with_pooling_results_and_derivative_and_update_weight(const real * input_V, integer nir, real * weight_M, real ** derivative_M, integer ndm, integer ndr, integer ndc, real ** output_M, integer ** pci, integer dpk, real alpha);

/*	Calcute weighted sum with pooling result and a derivative item is multiplied , also weights and biases are updated: T(weight_M) * input_V , result is saved in output_V. Note a bias is considered in the weight matrix.
*   This function is used to back propagation the error term from layer h[0] to layer Ctop.
*	Parameters :
*		input_V		: the input vector.
*		nir			: the number of input vector rows.
*		weight_M	: the weight matrix.
*		derivative_M: the derivative matrix.
*		ndm			: the number of derivative matrix.
*		ndr			: the number of derivative vector rows.
*		ndc			: the number of derivative vector columns.
*		output_M	: the result vector.
*		pci			: the pooling result vector.
*		dpk			: the number of pooling columns
*		alpha		: the learning rate.
*		lambda		: the regularization rate.
*	Return :
*		void
*/
void weighted_sum_error_term_with_pooling_results_and_derivative_and_update_weight_with_regularization(const real * input_V, integer nir, real * weight_M, real ** derivative_M, integer ndm, integer ndr, integer ndc, real ** output_M, integer ** pci, integer dpk, real alpha, real lambda);
/*  Calculate convolution with pooling result and accumulate to output_M : output_M += input_M ** weight_M.Note a bias is considered in the weight matrix.
*	This function is used to back propagation the error term from layer cl+1 to layer cl.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		ipci		: the pooling result vector of input matrix.
*		idpk		: the number pooling columns of input matrix
*		weight_M	: the weight matrix.
*		nwm			: the number of weight matrixes
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		derivative_M: the derivative matrix.
*		ndr			: the number of derivative matrix rows.
*		ndc			: the number of derivative matrix columns.
*		dpci		: the pooling result vector of derivative matrix.
*		ddpk		: the number pooling columns of derivative matrix
*		output_M	: the result matrix.
*		alpha		: the learning rate
*		lwm			£ºa temp matrix to calculte to weight derivative.
*	Return :
*		void
*/
void convolution_error_term_with_pooling_result_and_update_weight(real * input_M, integer nir, integer nic, integer * ipci, integer idpk, real ** weight_M, integer nwm, integer wsz, real ** derivative_M, integer ndr, integer ndc, integer ** dpci, integer ddpk, real ** output_M, real alpha, real * lwm);

/*  Calculate convolution with pooling result and accumulate to output_M : output_M += input_M ** weight_M.Note a bias is considered in the weight matrix.
*	This function is used to back propagation the error term from layer cl+1 to layer cl.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		ipci		: the pooling result vector of input matrix.
*		idpk		: the number pooling columns of input matrix
*		weight_M	: the weight matrix.
*		nwm			: the number of weight matrixes
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		derivative_M: the derivative matrix.
*		ndr			: the number of derivative matrix rows.
*		ndc			: the number of derivative matrix columns.
*		dpci		: the pooling result vector of derivative matrix.
*		ddpk		: the number pooling columns of derivative matrix
*		output_M	: the result matrix.
*		alpha		: the learning rate
*		lambda		: the regularization rate.
*		lwm			£ºa temp matrix to calculte to weight derivative.
*	Return :
*		void
*/
void convolution_error_term_with_pooling_result_and_update_weight_with_regularization(real * input_M, integer nir, integer nic, integer * ipci, integer idpk, real ** weight_M, integer nwm, integer wsz, real ** derivative_M, integer ndr, integer ndc, integer ** dpci, integer ddpk, real ** output_M, real alpha, real lambda, real * lwm);

/*  Calculate convolution and accumulate to output_M : output_M += input_M ** weight_M.Note a bias is considered in the weight matrix.
*	This function is used to back propagation the error term from layer cl0 to layer input.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		ipci		: the pooling result vector of input matrix.
*		idpk		: the number pooling columns of input matrix
*		weight_M	: the weight matrix.
*		nwm			: the number of weight matrixes
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		derivative_M: the derivative matrix.
*		ndr			: the number of derivative matrix rows.
*		ndc			: the number of derivative matrix columns.
*		dpci		: the pooling result vector of derivative matrix.
*		ddpk		: the number pooling columns of derivative matrix
*		output_M	: the result matrix.
*		alpha		: the learning rate
*		lwm			£ºa temp matrix to calculte to weight derivative.
*	Return :
*		void
*/
void convolution_error_term_and_update_weight(real * input_M, integer nir, integer nic, integer * ipci, integer idpk, real * weight_M, integer wsz, real * derivative_M, integer ndr, integer ndc, real * output_M, real alpha, real * lwm);

/*  Calculate convolution and accumulate to output_M : output_M += input_M ** weight_M.Note a bias is considered in the weight matrix.
*	This function is used to back propagation the error term from layer cl0 to layer input.
*	Parameters :
*		input_M		: the input matrix.
*		nir			: the number of input matrix rows.
*		nic			: the number of input matrix colums.
*		ipci		: the pooling result vector of input matrix.
*		idpk		: the number pooling columns of input matrix
*		weight_M	: the weight matrix.
*		nwm			: the number of weight matrixes
*		nwr			: the number of weight matrix rows.
*		wsz			: the number of weight matrix columns.
*		derivative_M: the derivative matrix.
*		ndr			: the number of derivative matrix rows.
*		ndc			: the number of derivative matrix columns.
*		dpci		: the pooling result vector of derivative matrix.
*		ddpk		: the number pooling columns of derivative matrix
*		output_M	: the result matrix.
*		alpha		: the learning rate
*		lambda		: the regularization rate.
*		lwm			£ºa temp matrix to calculte to weight derivative.
*	Return :
*		void
*/
void convolution_error_term_and_update_weight_with_regularization(real * input_M, integer nir, integer nic, integer * ipci, integer idpk, real * weight_M, integer wsz, real * derivative_M, integer ndr, integer ndc, real * output_M, real alpha, real lambda, real * lwm);
#endif 
