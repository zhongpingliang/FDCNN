#include "operator.h"
#include <cstdlib>
#include <cmath>
#include <cstring>


void convolution(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M){
	integer i1, i2, i3, i4, i5, wlb;
	const integer nrc = wsz + nic - 1;		//number of result matrix columns  
	const integer nwc = wsz * nir + 1;		//number of weight matrix columns  
	real z = 0;							//weight sum.
	for (i1 = 0; i1 < nrc; ++i1){
		for (i2 = 0; i2 < nwr; ++i2){
			i3 = i1;
			i4 = nwc - 2;
			if (i3 >= nic){
				i4 -= (i3 - nic + 1) * nir;
				i3 = nic - 1;
			}
			for (z = 0, wlb = i1 - wsz; i3 > -1 && i3 > wlb; --i3) for (i5 = nir - 1; i5 > -1; --i5, --i4) z += weight_M[i2 * nwc + i4] * input_M[i5 * nic + i3];
			output_M[i2 * nrc + i1] = z + weight_M[(1 + i2) * nwc - 1];//add bias
		}
	}
}

void convolution_and_non_linear(const real * input_M, integer nir, integer nic, const real * weight_M, integer nwr, integer wsz, real * output_M){
	integer i1, i2, i3, i4, i5, wlb;
	const integer nrc = wsz + nic - 1;		//number of result matrix columns  
	const integer nwc = wsz * nir + 1;		//number of weight matrix columns  
	real z = 0;							//weight sum.
	for (i1 = 0; i1 < nrc; ++i1){
		for (i2 = 0; i2 < nwr; ++i2){
			i3 = i1;
			i4 = nwc - 2;
			if (i3 >= nic){
				i4 -= (i3 - nic + 1) * nir;
				i3 = nic - 1;
			}
			for (z = 0, wlb = i1 - wsz; i3 > -1 && i3 > wlb; --i3) for (i5 = nir - 1; i5 > -1; --i5, --i4) z += weight_M[i2 * nwc + i4] * input_M[i5 * nic + i3];
			z += weight_M[(1 + i2) * nwc - 1];	//add bias
			output_M[i2 * nrc + i1] = non_linear_function(z);
		}
	}
}

void convolution_with_pooling_result(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M, integer * pci, integer dpk){
	integer i1, i2, i3, i4, i5, wlb;
	const integer nrc = wsz + dpk - 1;		//number of result matrix columns  
	const integer nwc = wsz * nir + 1;		//number of weight matrix columns  
	real z = 0;							//weight sum.
	for (i1 = 0; i1 < nrc; ++i1){
		for (i2 = 0; i2 < nwr; ++i2){
			i3 = i1;
			i4 = nwc - 2;
			if (i3 >= dpk){
				i4 -= (i3 - dpk + 1) * nir;
				i3 = dpk - 1;
			}
			for (z = 0, wlb = i1 - wsz; i3 > -1 && i3 > wlb; --i3) for (i5 = nir - 1; i5 > -1; --i5, --i4) z += weight_M[i2 * nwc + i4] * input_M[i5 * nic + pci[i3]];
			output_M[i2 * nrc + i1] = z + weight_M[(1 + i2) * nwc - 1];	//add bias
		}
	}
}

void convolution_with_pooling_result_and_non_linear(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M, integer * pci, integer dpk){
	integer i1, i2, i3, i4, i5, wlb;
	const integer nrc = wsz + dpk - 1;		//number of result matrix columns  
	const integer nwc = wsz * nir + 1;		//number of weight matrix columns  
	real z = 0;							//weight sum.
	for (i1 = 0; i1 < nrc; ++i1){
		for (i2 = 0; i2 < nwr; ++i2){
			i3 = i1;
			i4 = nwc - 2;
			if (i3 >= dpk){
				i4 -= (i3 - dpk + 1) * nir;
				i3 = dpk - 1;
			}
			for (z = 0, wlb = i1 - wsz; i3 > -1 && i3 > wlb; --i3) for (i5 = nir - 1; i5 > -1; --i5, --i4) z += weight_M[i2 * nwc + i4] * input_M[i5 * nic + pci[i3]];
			z += weight_M[(1 + i2) * nwc - 1];	//add bias
			output_M[i2 * nrc + i1] = non_linear_function(z);
		}
	}
}

void convolution_with_pooling_result_accumulate(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M, integer * pci, integer dpk)
{
	integer i1, i2, i3, i4, i5, wlb;
	const integer nrc = wsz + dpk - 1;		//number of result matrix columns  
	const integer nwc = wsz * nir + 1;		//number of weight matrix columns  
	real z = 0;							//weight sum.
	for (i1 = 0; i1 < nrc; ++i1){
		for (i2 = 0; i2 < nwr; ++i2){
			i3 = i1;
			i4 = nwc - 2;
			if (i3 >= dpk){
				i4 -= (i3 - dpk + 1) * nir;
				i3 = dpk - 1;
			}
			for (z = 0, wlb = i1 - wsz; i3 > -1 && i3 > wlb; --i3) for (i5 = nir - 1; i5 > -1; --i5, --i4) z += weight_M[i2 * nwc + i4] * input_M[i5 * nic + pci[i3]];
			output_M[i2 * nrc + i1] += z + weight_M[(1 + i2) * nwc - 1];	//add bias
		}
	}
}

void convolution_with_pooling_result_accumulate_and_non_linear(real * input_M, integer nir, integer nic, real * weight_M, integer nwr, integer wsz, real * output_M, integer * pci, integer dpk)
{
	integer i1, i2, i3, i4, i5, wlb;
	const integer nrc = wsz + dpk - 1;		//number of result matrix columns  
	const integer nwc = wsz * nir + 1;		//number of weight matrix columns  
	real z = 0;							//weight sum.
	for (i1 = 0; i1 < nrc; ++i1){
		for (i2 = 0; i2 < nwr; ++i2){
			i3 = i1;
			i4 = nwc - 2;
			if (i3 >= dpk){
				i4 -= (i3 - dpk + 1) * nir;
				i3 = dpk - 1;
			}
			for (z = 0, wlb = i1 - wsz; i3 > -1 && i3 > wlb; --i3) for (i5 = nir - 1; i5 > -1; --i5, --i4) z += weight_M[i2 * nwc + i4] * input_M[i5 * nic + pci[i3]];
			z += weight_M[(1 + i2) * nwc - 1] + output_M[i2 * nrc + i1];	//add bias
			output_M[i2 * nrc + i1] = non_linear_function(z);
		}
	}
}

void weighted_sum(const real * input_V, integer nir, const real * weight_M, integer nwr, real * output_V){
	const integer nwc = nir + 1;		//number of weight matrix columns
	integer i1, i2;
	real z = 0;
	for (i1 = 0; i1 < nwr; ++i1){
		for (z = 0, i2 = 0; i2 < nir; ++i2) z += weight_M[i1* nwc + i2] * input_V[i2];
		output_V[i1] = z + weight_M[i1 * nwc + nir];	//add bias
	}
}

void weighted_sum_and_non_linear(const real * input_V, integer nir, const real * weight_M, integer nwr, real * output_V){
	const integer nwc = nir + 1;		//number of weight matrix columns
	integer i1, i2;
	real z = 0;
	for (i1 = 0; i1 < nwr; ++i1){
		for (z = 0, i2 = 0; i2 < nir; ++i2) z += weight_M[i1* nwc + i2] * input_V[i2];
		z += weight_M[i1 * nwc + nir];			//add bias
		output_V[i1] = non_linear_function(z);
	}
}

void weighted_sum_with_pooling_results(real ** input_M, integer nim, integer nir, integer nic, const real * weight_M, integer nwr, real * output_V, integer ** pci, integer dpk){
	const integer tmp = dpk * nir;
	const integer nwc = nim * tmp + 1;		//number of weight matrix columns
	integer i1, i2, i3, i4;
	real z = 0;
	for (i1 = 0; i1 < nwr; ++i1){
		for (z = 0, i2 = 0; i2 < nim; ++i2)for (i3 = 0; i3 < dpk; ++i3) for (i4 = 0; i4 < nir; ++i4) z += weight_M[i1 * nwc + i2 * tmp + i3 * nir + i4] * input_M[i2][i4 * nic + pci[i2][i3]];
		output_V[i1] = z + weight_M[(i1 + 1) * nwc - 1];	//add bias
	}
}

void weighted_sum_with_pooling_results_and_non_linear(real ** input_M, integer nim, integer nir, integer nic, const real * weight_M, integer nwr, real * output_V, integer ** pci, integer dpk){
	const integer tmp = dpk * nir;
	const integer nwc = nim * tmp + 1;		//number of weight matrix columns
	integer i1, i2, i3, i4;
	real z = 0;
	for (i1 = 0; i1 < nwr; ++i1){
		for (z = 0, i2 = 0; i2 < nim; ++i2) for (i3 = 0; i3 < dpk; ++i3) for (i4 = 0; i4 < nir; ++i4) z += weight_M[i1 * nwc + i2 * tmp + i3 * nir + i4] * input_M[i2][i4 * nic + pci[i2][i3]];
		z += weight_M[(i1 + 1) * nwc - 1];			//add bias
		output_V[i1] = non_linear_function(z);
	}
}

void weighted_sum_error_term_and_derivative(const real * input_V, integer nir, const real * weight_M, real * derivative_V, integer ndr, real * output_V){
	const integer nwc = ndr + 1;		//number of weight matrix columns
	integer i1, i2;
	real z = 0;
	for (i1 = 0; i1 < ndr; ++i1){
		for (z = 0, i2 = 0; i2 < nir; ++i2) z += weight_M[i2 * nwc + i1] * input_V[i2];
		output_V[i1] = z * non_linear_function_derivative(derivative_V[i1]);
	}
}

void weighted_sum_error_term_and_derivative_and_update_weight(const real * input_V, integer nir, real * weight_M, real * derivative_V, integer ndr, real * output_V, real alpha){
	const integer nwc = ndr + 1;		//number of weight matrix columns
	integer i1, i2;
	real z = 0;
	for (i1 = 0; i1 < ndr; ++i1){
		for (z = 0, i2 = 0; i2 < nir; ++i2) {
			z += weight_M[i2 * nwc + i1] * input_V[i2];
			weight_M[i2 * nwc + i1] -= alpha * input_V[i2] * derivative_V[i1];		//update weight
		}
		output_V[i1] = z * non_linear_function_derivative(derivative_V[i1]);
	}
	for (i2 = 0; i2 < nir; ++i2) weight_M[i2 * nwc + ndr] -= alpha * input_V[i2];	// update bias
}

void weighted_sum_error_term_and_derivative_and_update_weight_with_regularization(const real * input_V, integer nir, real * weight_M, real * derivative_V, integer ndr, real * output_V, real alpha, real lambda){
	const integer nwc = ndr + 1;		//number of weight matrix columns
	integer i1, i2;
	real z = 0;
	for (i1 = 0; i1 < ndr; ++i1){
		for (z = 0, i2 = 0; i2 < nir; ++i2) {
			z += weight_M[i2 * nwc + i1] * input_V[i2];
			weight_M[i2 * nwc + i1] -= alpha * input_V[i2] * derivative_V[i1] + lambda * weight_M[i2 * nwc + i1];		//update weight
		}
		output_V[i1] = z * non_linear_function_derivative(derivative_V[i1]);
	}
	for (i2 = 0; i2 < nir; ++i2) weight_M[i2 * nwc + ndr] -= alpha * input_V[i2] + lambda * weight_M[i2 * nwc + ndr];	// update bias
}

void weighted_sum_error_term_with_pooling_results_and_derivative_and_update_weight(const real * input_V, integer nir, real * weight_M, real ** derivative_M, integer ndm, integer ndr, integer ndc, real ** output_M, integer ** pci, integer dpk, real alpha){
	const integer tmp = dpk * ndr;
	const integer nwc = ndm * tmp + 1;
	integer i1, i2, i3, i4, i5, i6;
	real z = 0;
	for (i1 = 0; i1 < ndm; ++i1) for (i2 = 0; i2 < dpk; ++i2) for (i3 = 0; i3 < ndr; ++i3){
		i5 = i3 * ndc + pci[i1][i2];
		for (z = 0, i4 = 0; i4 < nir; ++i4){
			i6 = i4 * nwc + i1 * tmp + i2 * ndr + i3;
			z += weight_M[i6] * input_V[i4];
			weight_M[i6] -= alpha * input_V[i4] * derivative_M[i1][i5];	//update weight
		}
		output_M[i1][i5] = z * non_linear_function_derivative(derivative_M[i1][i5]);
	}
	for (i4 = 0; i4 < nir; ++i4) weight_M[(i4 + 1) * nwc - 1] -= alpha * input_V[i4];	// update bias
}

void weighted_sum_error_term_with_pooling_results_and_derivative_and_update_weight_with_regularization(const real * input_V, integer nir, real * weight_M, real ** derivative_M, integer ndm, integer ndr, integer ndc, real ** output_M, integer ** pci, integer dpk, real alpha, real lambda){
	const integer tmp = dpk * ndr;
	const integer nwc = ndm * tmp + 1;
	integer i1, i2, i3, i4, i5, i6;
	real z = 0;
	for (i1 = 0; i1 < ndm; ++i1) for (i2 = 0; i2 < dpk; ++i2) for (i3 = 0; i3 < ndr; ++i3){
		i5 = i3 * ndc + pci[i1][i2];
		for (z = 0, i4 = 0; i4 < nir; ++i4){
			i6 = i4 * nwc + i1 * tmp + i2 * ndr + i3;
			z += weight_M[i6] * input_V[i4];
			weight_M[i6] -= alpha * input_V[i4] * derivative_M[i1][i5] + lambda * weight_M[i6];	//update weight
		}
		output_M[i1][i5] = z * non_linear_function_derivative(derivative_M[i1][i5]);
	}
	for (i4 = 0; i4 < nir; ++i4) weight_M[(i4 + 1) * nwc - 1] -= alpha * input_V[i4] + lambda * weight_M[(i4 + 1) * nwc - 1];	// update bias
}

void convolution_error_term_with_pooling_result_and_update_weight(real * input_M, integer nir, integer nic, integer * ipci, integer idpk, real ** weight_M, integer nwm, integer wsz, real ** derivative_M, integer ndr, integer ndc, integer ** dpci, integer ddpk, real ** output_M, real alpha, real * lwm){
	const integer nwc = wsz * ndr + 1;
	const integer tmp = nwc * nir;
	integer i1, i2, i3, i4, i5, i6, wlb;
	real z = 0;
	for (i1 = 0; i1 < nwm; ++i1) {
		memset(lwm, 0, sizeof(real)* tmp);
		for (i2 = 0; i2 < idpk; ++i2){
			i3 = ipci[i2];
			i4 = nwc - 2;
			if (i3 >= ddpk){
				i4 -= (i3 - ddpk + 1) * ndr;
				i3 = ddpk - 1;
			}
			for (wlb = ipci[i2] - wsz; i3 > -1 && i3 > wlb; --i3) for (i5 = ndr - 1; i5 > -1; --i5, --i4) {
				for (z = 0, i6 = 0; i6 < nir; ++i6){
					lwm[i6 * nwc + i4] += input_M[i6 * nic + ipci[i2]] * derivative_M[i1][i5 * ndc + dpci[i1][i3]];	//weight
					z += input_M[i6 * nic + ipci[i2]] * weight_M[i1][i6 * nwc + i4];
				}
				output_M[i1][i5 * ndc + dpci[i1][i3]] += z;
			}
			for (i6 = 0; i6 < nir; ++i6) lwm[(i6 + 1) * nwc - 1] += input_M[i6 * nic + ipci[i2]];	//bias
		}
		for (i2 = 0; i2 < tmp; ++i2) weight_M[i1][i2] -= alpha * lwm[i2];
	}
}

void convolution_error_term_with_pooling_result_and_update_weight_with_regularization(real * input_M, integer nir, integer nic, integer * ipci, integer idpk, real ** weight_M, integer nwm, integer wsz, real ** derivative_M, integer ndr, integer ndc, integer ** dpci, integer ddpk, real ** output_M, real alpha, real lambda, real * lwm){
	const integer nwc = wsz * ndr + 1;
	const integer tmp = nwc * nir;
	integer i1, i2, i3, i4, i5, i6, wlb;
	real z = 0;
	for (i1 = 0; i1 < nwm; ++i1) {
		memset(lwm, 0, sizeof(real)* tmp);
		for (i2 = 0; i2 < idpk; ++i2){
			i3 = ipci[i2];
			i4 = nwc - 2;
			if (i3 >= ddpk){
				i4 -= (i3 - ddpk + 1) * ndr;
				i3 = ddpk - 1;
			}
			for (wlb = ipci[i2] - wsz; i3 > -1 && i3 > wlb; --i3) for (i5 = ndr - 1; i5 > -1; --i5, --i4) {
				for (z = 0, i6 = 0; i6 < nir; ++i6){
					lwm[i6 * nwc + i4] += input_M[i6 * nic + ipci[i2]] * derivative_M[i1][i5 * ndc + dpci[i1][i3]];	//weight
					z += input_M[i6 * nic + ipci[i2]] * weight_M[i1][i6 * nwc + i4];
				}
				output_M[i1][i5 * ndc + dpci[i1][i3]] += z;
			}
			for (i6 = 0; i6 < nir; ++i6) lwm[(i6 + 1) * nwc - 1] += input_M[i6 * nic + ipci[i2]];	//bias
		}
		for (i2 = 0; i2 < tmp; ++i2) weight_M[i1][i2] -= alpha * lwm[i2] + lambda * weight_M[i1][i2];
	}
}

void convolution_error_term_and_update_weight(real * input_M, integer nir, integer nic, integer * ipci, integer idpk, real * weight_M, integer wsz, real * derivative_M, integer ndr, integer ndc, real * output_M, real alpha, real * lwm){
	const integer nwc = wsz * ndr + 1;
	const integer tmp = nwc * nir;
	integer i1, i2, i3, i4, i5, wlb;
	real z = 0;
	memset(lwm, 0, sizeof(real)* tmp);
	for (i1 = 0; i1 < idpk; ++i1){
		i2 = ipci[i1];
		i3 = nwc - 2;
		if (i2 >= ndc){
			i3 -= (i2 - ndc + 1) * ndr;
			i2 = ndc - 1;
		}
		for (wlb = ipci[i1] - wsz; i2 > -1 && i2 > wlb; --i2) for (i4 = ndr - 1; i4 > -1; --i4, --i3){
			for (z = 0, i5 = 0; i5 < nir; ++i5){
				lwm[i5 * nwc + i3] += input_M[i5 * nic + ipci[i1]] * derivative_M[i4 * ndc + i2]; //weight
				z += input_M[i5 * nic + ipci[i1]] * weight_M[i5 * nwc + i3];
			}
			output_M[i4 * ndc + i2] += z;
		}
		for (i5 = 0; i5 < nir; ++i5) lwm[(i5 + 1) * nwc - 1] += input_M[i5 * nic + ipci[i1]];	//bias
	}
	for (i1 = 0; i1 < tmp; ++i1) weight_M[i1] -= alpha * lwm[i1];
}

void convolution_error_term_and_update_weight_with_regularization(real * input_M, integer nir, integer nic, integer * ipci, integer idpk, real * weight_M, integer wsz, real * derivative_M, integer ndr, integer ndc, real * output_M, real alpha, real lambda, real * lwm){
	const integer nwc = wsz * ndr + 1;
	const integer tmp = nwc * nir;
	integer i1, i2, i3, i4, i5, wlb;
	real z = 0;
	memset(lwm, 0, sizeof(real)* tmp);
	for (i1 = 0; i1 < idpk; ++i1){
		i2 = ipci[i1];
		i3 = nwc - 2;
		if (i2 >= ndc){
			i3 -= (i2 - ndc + 1) * ndr;
			i2 = ndc - 1;
		}
		for (wlb = ipci[i1] - wsz; i2 > -1 && i2 > wlb; --i2) for (i4 = ndr - 1; i4 > -1; --i4, --i3){
			for (z = 0, i5 = 0; i5 < nir; ++i5){
				lwm[i5 * nwc + i3] += input_M[i5 * nic + ipci[i1]] * derivative_M[i4 * ndc + i2]; //weight
				z += input_M[i5 * nic + ipci[i1]] * weight_M[i5 * nwc + i3];
			}
			output_M[i4 * ndc + i2] += z;
		}
		for (i5 = 0; i5 < nir; ++i5) lwm[(i5 + 1) * nwc - 1] += input_M[i5 * nic + ipci[i1]];	//bias
	}
	for (i1 = 0; i1 < tmp; ++i1) weight_M[i1] -= alpha * lwm[i1] + lambda * weight_M[i1];
}

void k_max_pooling(const real * input_M, integer nir, integer nic, integer * output_V, integer k){
	integer i1, i2;
	real z = 0;
	real dpa[MAX_SEN_LEN];		//save the lengths of max vectors in up order.
	for (i1 = 0; i1 < k; ++i1) dpa[i1] = -1;
	for (i1 = 0; i1 < nic; ++i1){
		for (z = 0, i2 = 0; i2 < nir; ++i2)	z += square(input_M[i2 * nic + i1]);
		for (i2 = k - 2; i2 > -1; --i2){
			if (z > dpa[i2]) {
				dpa[i2 + 1] = dpa[i2];
				output_V[i2 + 1] = output_V[i2];
			}
			else break;
		}
		if (dpa[i2 + 1] < z) {
			dpa[i2 + 1] = z;
			output_V[i2 + 1] = i1;
		}
	}
	qsort(output_V, k, sizeof(integer), smaller);
}