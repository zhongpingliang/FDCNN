#include "sentence_model_base.h"
#include <thread>
#include <vector>
#include <cstring>

using std::pair;
using std::thread;
using std::vector;

SentenceModelBase::SentenceModelBase()
: nwd(0)
, ncl(0)
, scw(0)
, ncr(0)
, MM(0)
, ncm(0)
, Ktop(0)
, nhl(0)
, nhr(0)
, WM(0)
, U(0)
, nor(0)
, alp(0.01)
, lmd(0)
, nfd(0)
, nif(0)
, nte(0)
, feature_table(0)
, iter_num(1000)
, thrd_num(1)
, train_line_num(0)
, exa_num(0)
, avgerr(0)
, dicts(0)
, linu(0)
, citer(1)
, bin(1)
, train_fin(0)
, pred_fin(0)
, pred_fout(0)
, model_file(0)
, config_file(0)
{
	config_file_name[0] = 0;
	train_file_name[0] = 0;
	model_file_name[0] = 0;
	predict_in_file_name[0] = 0;
	predict_out_file_name[0] = 0;
	word2vec_file_name[0] = 0;
}

SentenceModelBase::SentenceModelBase(const SentenceModelBase & instance){}

SentenceModelBase & SentenceModelBase::operator=(const SentenceModelBase & instance){
	return *this;
}

real SentenceModelBase::get_rand(integer layer_size){
	return (rand() / (real)RAND_MAX - 0.5) / sqrt(layer_size);
	//return (rand() / (real)RAND_MAX - 0.5) / layer_size;
}

void SentenceModelBase::read_config(){
	fprintf(stdout, "Start reading config\n");
	integer i;
	fscanf(config_file, "nif = %d\n", &nif);
	nfd = (integer *)malloc(sizeof(integer)* nif);
	fscanf(config_file, "nfd = ");
	nwd = 0;
	for (i = 0; i < nif; ++i) {
		fscanf(config_file, "%d ", &nfd[i]);
		nwd += nfd[i];
	}
	fscanf(config_file, "ncl = %d\n", &ncl);
	scw = (integer *)malloc(sizeof(integer)* ncl);
	ncr = (integer *)malloc(sizeof(integer)* ncl);
	ncm = (integer *)malloc(sizeof(integer)* ncl);
	fscanf(config_file, "scw = ");
	for (i = 0; i < ncl; ++i) fscanf(config_file, "%d ", &scw[i]);
	fscanf(config_file, "\nncr = ");
	for (i = 0; i < ncl; ++i) fscanf(config_file, "%d ", &ncr[i]);
	fscanf(config_file, "\nncm = ");
	for (i = 0; i < ncl; ++i) fscanf(config_file, "%d ", &ncm[i]);
	fscanf(config_file, "\nKtop = %d", &Ktop);
	fscanf(config_file, "\nnhl = %d", &nhl);
	nhr = (integer *)malloc(sizeof(integer)* nhl);
	fscanf(config_file, "\nnhr = ");
	for (i = 0; i < nhl; ++i) fscanf(config_file, "%d ", &nhr[i]);
	fscanf(config_file, "\n");

	fprintf(stdout, "\tnif = %d\n", nif);
	fprintf(stdout, "\tnfd = ");
	for (i = 0; i < nif; ++i) fprintf(stdout, "%d ", nfd[i]);
	fprintf(stdout, "\n\tnwd = %d\n", nwd);
	fprintf(stdout, "\tncl = %d\n", ncl);
	fprintf(stdout, "\tscw = ");
	for (i = 0; i < ncl; ++i) fprintf(stdout, "%d ", scw[i]);
	fprintf(stdout, "\n\tncr = ");
	for (i = 0; i < ncl; ++i) fprintf(stdout, "%d ", ncr[i]);
	fprintf(stdout, "\n\tncm = ");
	for (i = 0; i < ncl; ++i) fprintf(stdout, "%d ", ncm[i]);
	fprintf(stdout, "\n\tKtop = %d", Ktop);
	fprintf(stdout, "\n\tnhl = %d", nhl);
	fprintf(stdout, "\n\tnhr = ");
	for (i = 0; i < nhl; ++i) fprintf(stdout, "%d ", nhr[i]);
	fprintf(stdout, "\n");
	fprintf(stdout, "Finish reading config\n");
}

void SentenceModelBase::check_config(){
	integer i, flag = 0;
	if (nif < 1){
		fprintf(stderr, "Number of features can not be small than 1.\n");
		flag = 1;
	}
	if (nif > MAX_FIELD_NUM){
		fprintf(stderr, "Number of features can not exceed MAX_FIRLD_NUM = %d.\n", MAX_FIELD_NUM);
		flag = 1;
	}
	for (i = 0; i < nif; ++i) if (nfd[i] < 1){
		fprintf(stderr, "Dimension of the %dth features can not be small than 1.\n", i + 1);
		flag = 1;
	}
	if (ncl < 1){
		fprintf(stderr, "Number of convolution layers can not be small than 1.\n");
		flag = 1;
	}
	for (i = 0; i < ncl; ++i) if (scw[i] < 1){
		fprintf(stderr, "Size of window in the %dth layer can not be small than 1.\n", i + 1);
		flag = 1;
	}
	for (i = 0; i < ncl; ++i) if (ncr[i] < 1) {
		fprintf(stderr, "Dimension of rows in the %dth convolution layer can not be small than 1.\n", i + 1);
		flag = 1;
	}
	for (i = 0; i < ncl; ++i) if (ncm[i] < 1){
		fprintf(stderr, "Number of convolutions in the %dth convolution layer can not be small than 1.\n", i + 1);
		flag = 1;
	}
	if (Ktop < 1) {
		fprintf(stderr, "Ktop can not be small than 1.\n");
		flag = 1;
	}
	if (nhl < 1) {
		fprintf(stderr, "Number of hidden layers can not be small than 1.\n");
		flag = 1;
	}
	for (i = 0; i < nhl; ++i) if (nhr[i] < 1) {
		fprintf(stderr, "Dimension of rows in the %dth hidden layer can not be small than 1.\n", i + 1);
		flag = 1;
	}
	if (flag) exit(1);
}

void SentenceModelBase::alloc_mem(){
	integer i, j, tmp;
	MM = (real ***)malloc(ncl * sizeof(real**));
	MM[0] = (real **)malloc(ncm[0] * sizeof(real*));
	for (j = 0; j < ncm[0]; ++j) MM[0][j] = (real *)malloc(ncr[0] * (scw[0] * nwd + 1) * sizeof(real));
	for (i = 1; i < ncl; ++i) {
		tmp = ncm[i - 1] * ncm[i];
		MM[i] = (real **)malloc(tmp * sizeof(real*));
		for (j = 0; j < tmp; ++j) MM[i][j] = (real*)malloc(ncr[i] * (scw[i] * ncr[i - 1] + 1) * sizeof(real));
	}
	WM = (real **)malloc(nhl * sizeof(real *));
	WM[0] = (real *)malloc(nhr[0] * (Ktop * ncr[ncl - 1] * ncm[ncl - 1] + 1)  * sizeof(real));
	for (i = 1; i < nhl; ++i) WM[i] = (real *)malloc(nhr[i] * (nhr[i - 1] + 1) * sizeof(real));
	U = (real *)malloc(nor * (nhr[nhl - 1] + 1) * sizeof(real));
	if (word2vec_file_name[0]) {
		for (i = 1; i < nif; ++i) feature_table[i] = (real*)malloc(dicts[i].size() *  nfd[i] * sizeof(real));
	} else {
		feature_table = (real **)malloc(nif * sizeof(real*));
		for (i = 0; i < nif; ++i) feature_table[i] = (real*)malloc(dicts[i].size() *  nfd[i] * sizeof(real));
	}
}

void SentenceModelBase::free_men(){
	integer i, j, tmp;
	for (j = 0; j < ncm[0]; ++j) free(MM[0][j]);
	free(MM[0]);
	for (i = 1; i < ncl; ++i) {
		tmp = ncm[i - 1] * ncm[i];
		for (j = 0; j < tmp; ++j) free(MM[i][j]);
		free(MM[i]);
	}
	free(MM);
	for (i = 0; i < nhl; ++i) free(WM[i]);
	free(WM);
	free(U);
	for (i = 0; i < nif; ++i) free(feature_table[i]);
	free(feature_table);
	free(nfd);
	free(scw);
	free(ncr);
	free(ncm);
	free(nhr);
	delete[]dicts;
}

void SentenceModelBase::init(){
	integer i, j, k, tmp1, tmp2, layer_size;
	layer_size = scw[0] * nwd + 1;
	tmp2 = ncr[0] * layer_size;
	for (j = 0; j < ncm[0]; ++j) for (k = 0; k < tmp2; ++k)	MM[0][j][k] = get_rand(layer_size);
	for (i = 1; i < ncl; ++i) {
		tmp1 = ncm[i - 1] * ncm[i];
		layer_size = scw[i] * ncr[i - 1] + 1;
		tmp2 = ncr[i] * layer_size;
		for (j = 0; j < tmp1; ++j) for (k = 0; k < tmp2; ++k) MM[i][j][k] = get_rand(layer_size);
	}
	layer_size = Ktop * ncr[ncl - 1] * ncm[ncl - 1] + 1;
	tmp1 = nhr[0] * layer_size;
	for (j = 0; j < tmp1; ++j) WM[0][j] = get_rand(layer_size);
	for (i = 1; i < nhl; ++i) {
		layer_size = nhr[i - 1] + 1;
		tmp1 = nhr[i] * layer_size;
		for (j = 0; j < tmp1; ++j) WM[i][j] = get_rand(layer_size);
	}
	layer_size = nhr[nhl - 1] + 1;
	tmp1 = nor * layer_size;
	for (j = 0; j < tmp1; ++j) U[j] = get_rand(layer_size);
	if (word2vec_file_name[0]) i = 1;
	else i = 0;
	for (; i < nif; ++i) {
		for (k = 0; k < nfd[i]; ++k) feature_table[i][k] = 0;
		for (j = 1; j < dicts[i].size(); ++j) for (k = 0; k < nfd[i]; ++k) feature_table[i][j * nfd[i] + k] = get_rand(nfd[i]);
	}
}

integer SentenceModelBase::dynamic_k(integer sen_len, integer cncl){
	return max(Ktop, sen_len + ((real)(Ktop - sen_len)) / ncl * (cncl + 1));
}

void SentenceModelBase::forward(real * SENM, integer lst, real***CM, integer * ncc, integer ***kpv, integer *ndk, real ** hve, real *ove){
	integer i1, i2, i3;
	integer tmp = 0, ran = 0;
	//sentence layer to convolution and pooling layers. 
	ncc[0] = lst + scw[0] - 1;
	for (i1 = 0; i1 < ncm[0]; ++i1) {
		convolution_and_non_linear(SENM, nwd, lst, MM[0][i1], ncr[0], scw[0], CM[0][i1]);
		k_max_pooling(CM[0][i1], ncr[0], ncc[0], kpv[0][i1], ndk[0]);
	}
	for (i1 = 1; i1 < ncl; ++i1){
		tmp = i1 - 1;
		ncc[i1] = ndk[tmp] + scw[i1] - 1;
		if (ncm[tmp] == 1) for (i2 = 0; i2 < ncm[i1]; ++i2){
			convolution_with_pooling_result_and_non_linear(CM[tmp][0], ncr[tmp], ncc[tmp], MM[i1][i2], ncr[i1], scw[i1], CM[i1][i2], kpv[tmp][0], ndk[tmp]);
			k_max_pooling(CM[i1][i2], ncr[i1], ncc[i1], kpv[i1][i2], ndk[i1]);
		}
		else for (i2 = 0; i2 < ncm[i1]; ++i2){
			convolution_with_pooling_result(CM[tmp][0], ncr[tmp], ncc[tmp], MM[i1][i2*ncm[tmp]], ncr[i1], scw[i1], CM[i1][i2], kpv[tmp][0], ndk[tmp]);
			ran = ncm[tmp] - 1;
			for (i3 = 1; i3 < ran; ++i3) convolution_with_pooling_result_accumulate(CM[tmp][i3], ncr[tmp], ncc[tmp], MM[i1][i2*ncm[tmp] + i3], ncr[i1], scw[i1], CM[i1][i2], kpv[tmp][i3], ndk[tmp]);
			convolution_with_pooling_result_accumulate_and_non_linear(CM[tmp][ran], ncr[tmp], ncc[tmp], MM[i1][i2*ncm[tmp] + ran], ncr[i1], scw[i1], CM[i1][i2], kpv[tmp][ran], ndk[tmp]);
			k_max_pooling(CM[i1][i2], ncr[i1], ncc[i1], kpv[i1][i2], ndk[i1]);
		}
	}
	//pooling to 0 hidden layer
	tmp = ncl - 1;
	if (ncl > 1) ncc[tmp] = ndk[tmp - 1] + scw[tmp] - 1;
	weighted_sum_with_pooling_results_and_non_linear(CM[tmp], ncm[tmp], ncr[tmp], ncc[tmp], WM[0], nhr[0], hve[0], kpv[tmp], ndk[tmp]);
	//hidden 0 to hidden top
	for (i1 = 1, tmp = i1 - 1; i1 < nhl; ++i1, ++tmp) weighted_sum_and_non_linear(hve[tmp], nhr[tmp], WM[i1], nhr[i1], hve[i1]);
	//hidden top to out layer
	tmp = nhl - 1;
	weighted_sum(hve[tmp], nhr[tmp], U, nor, ove);
	//some operation is need to the output layer, such as softmax.
}

void SentenceModelBase::backward(real * SENM, real *lsz, integer lst, real ***CM, real ***lcz, integer *ncc, integer ***kpv, integer *ndk, real **hve, real**lhz, real * loz, real * lche){
	const integer htop = nhl - 1;
	const integer ctop = ncl - 1;
	integer i1, i2, i3, tmp, ran;
	//some operation is need to the output layer, such as softmax.
	//out to hidden top
	weighted_sum_error_term_and_derivative_and_update_weight_with_regularization(loz, nor, U, hve[htop], nhr[htop], lhz[htop], alp, lmd);
	//hidden top to hidden 0
	for (i1 = htop, tmp = i1 - 1; i1 > 0; --i1, --tmp) weighted_sum_error_term_and_derivative_and_update_weight_with_regularization(lhz[i1], nhr[i1], WM[i1], hve[tmp], nhr[tmp], lhz[tmp], alp, lmd);
	//hidden 0 to cm top
	weighted_sum_error_term_with_pooling_results_and_derivative_and_update_weight_with_regularization(lhz[0], nhr[0], WM[0], CM[ctop], ncm[ctop], ncr[ctop], ncc[ctop], lcz[ctop], kpv[ctop], ndk[ctop], alp, lmd);
	//cm top to cm 0
	for (i1 = ctop, tmp = i1 - 1; i1 > 0; --i1, --tmp){
		//init lcz[i1 -1]* to 0
		for (i2 = 0; i2 < ncm[tmp]; ++i2) memset(lcz[tmp][i2], 0, sizeof(real)* ncr[tmp] * ncc[tmp]);
		//accumulate error term and update weight
		for (i2 = 0; i2 < ncm[i1]; ++i2) convolution_error_term_with_pooling_result_and_update_weight_with_regularization(lcz[i1][i2], ncr[i1], ncc[i1], kpv[i1][i2], ndk[i1], MM[i1] + i2 * ncm[tmp], ncm[tmp], scw[i1], CM[tmp], ncr[tmp], ncc[tmp], kpv[tmp], ndk[tmp], lcz[tmp], alp, lmd, lche);
		//act a derivative
		for (i2 = 0, ran = ncr[tmp] * ncc[tmp]; i2 < ncm[tmp]; ++i2) for (i3 = 0; i3 < ran; ++i3) lcz[tmp][i2][i3] *= non_linear_function_derivative(CM[tmp][i2][i3]);
	}
	// cm 0 to sen
	memset(lsz, 0, sizeof(real)* nwd * lst);
	for (i2 = 0; i2 < ncm[0]; ++i2)	convolution_error_term_and_update_weight_with_regularization(lcz[0][i2], ncr[0], ncc[0], kpv[0][i2], ndk[0], MM[0][i2], scw[0], SENM, nwd, lst, lsz, alp, lmd, lche);
	//word vector is not updated
}

void SentenceModelBase::look_up_table(map<string, integer>::iterator **features, real *SENM, integer lst){
	integer i, j, k, l;
	for (i = 0; i < lst; ++i) for (j = 0, k = 0; j < nif; k += nfd[j], ++j) for (l = 0; l < nfd[j]; ++l) SENM[(l + k) * lst + i] = feature_table[j][features[j][i]->second * nfd[j] + l];
}

void SentenceModelBase::update_feature_table_with_regularization(map<string, integer>::iterator **features, real *lsz, integer lst){
	integer i, j, k, l;
	for (i = 0; i < lst; ++i) {
		if (word2vec_file_name[0]) j = 1;
		else j = 0;
		for (k = 0; j < nif; k += nfd[j], ++j)if (features[j][i]->first != NIL)for (l = 0; l < nfd[j]; ++l) feature_table[j][features[j][i]->second * nfd[j] + l] -= alp * lsz[(l + k) * lst + i] + lmd * feature_table[j][features[j][i]->second * nfd[j] + l];
	}
}

integer SentenceModelBase::get_arg_pos(char *str, integer argc, char **argv) {
	integer a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			fprintf(stdout, "Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}
