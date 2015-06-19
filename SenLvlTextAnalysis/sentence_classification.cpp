#include "sentence_classification.h"
#include <thread>
#include <vector>
#include <cstring>

using std::pair;
using std::thread;
using std::vector;

SentenceClassification::SentenceClassification(){}

SentenceClassification::SentenceClassification(const SentenceClassification & instance){}

void SentenceClassification::make_dict(){
	fprintf(stdout, "Start making dictionary\n");
	char buffer[MAX_WORD_LEN], ch = 0;
	integer i = 0, j = 0, nflds = 0;
	string fld;
	const integer ndicts = 1 + nif;
	integer ids[MAX_FIELD_NUM + 1];
	dicts = new map<string, integer>[ndicts];
	memset(ids, 0, sizeof(integer)* (ndicts));
	for (i = 0; i < nif; ++i) dicts[i].insert(pair<string, integer>(NIL, ids[i]++));
	while (!feof(train_fin)){
		while (!feof(train_fin)) {
			ch = fgetc(train_fin);
			if (ch == '\n') ++train_line_num;
			else if (ch != '\r'){
				ungetc(ch, train_fin);
				break;
			}
		}
		if (feof(train_fin)) break;
		do{
			j = 0;
			for (i = 0; !feof(train_fin);){
				ch = fgetc(train_fin);
				if (ch == '\n') {
					++train_line_num;
					buffer[i] = 0;
					if (!buffer[0]) {
						fprintf(stderr, "Error while reading training file at %lld line. Unexpected new line is read.\n", train_line_num);
						exit(1);
					}
					fld.assign(buffer, i);
					if (dicts[nif].find(fld) == dicts[nif].end()) dicts[nif].insert(pair<string, integer>(fld, ids[nif]++));
					j = 1;
					break;
				}
				else if (ch == '\t'){
					buffer[i] = 0;
					break;
				}
				else if (ch != '\r'){
					if (i == MAX_WORD_LEN - 2){
						fprintf(stderr, "Error while reading training file at %lld line. There is a word with too long length.\n", train_line_num);
						exit(1);
					}
					buffer[i++] = ch;
				}
			}
			if (feof(train_fin)) {
				buffer[i] = 0;
				if (!buffer[0]) {
					fprintf(stderr, "Error while reading training file at %lld line. There is no label for the last example.\n", train_line_num);
					exit(1);
				}
				fld.assign(buffer, i);
				if (dicts[nif].find(fld) == dicts[nif].end()) dicts[nif].insert(pair<string, integer>(fld, ids[nif]++));
				j = 1;
			}
			if (j) break;
			for (nflds = 0, i = 0; !feof(train_fin);){
				ch = fgetc(train_fin);
				if (ch == '\n') {
					++train_line_num;
					if (nflds + 1 != nif){
						fprintf(stderr, "Error while reading training file at %lld line. Number of fields is not right.\n", train_line_num);
						exit(1);
					}
					buffer[i] = 0;
					fld.assign(buffer, i);
					if (dicts[nflds].find(fld) == dicts[nflds].end()) dicts[nflds].insert(pair<string, integer>(fld, ids[nflds]++));
					break;
				}
				else if (ch == '\t'){
					if (nflds + 1 >= nif){
						fprintf(stderr, "Error while reading training file at %lld line. Number of fields is not right.\n", train_line_num);
						exit(1);
					}
					buffer[i] = 0;
					fld.assign(buffer, i);
					if (dicts[nflds].find(fld) == dicts[nflds].end()) dicts[nflds].insert(pair<string, integer>(fld, ids[nflds]++));
					++nflds;
					i = 0;
				}
				else if (ch != '\r') {
					if (i == MAX_WORD_LEN - 2){
						fprintf(stderr, "Error while reading training file at %lld line. There is a word with too long length.\n", train_line_num);
						exit(1);
					}
					buffer[i++] = ch;
				}
			}
		} while (!feof(train_fin));
	}
	nor = dicts[nif].size();
	fprintf(stdout, "Finish making dictionary. there are %lld lines in train file.\n", train_line_num);
}

void SentenceClassification::make_dict_using_word2vec()
{
	fprintf(stdout, "Start making dictionary\n");
	char buffer[MAX_WORD_LEN], ch = 0;
	integer i = 0, j = 0, nflds = 0;
	string fld;
	const integer ndicts = 1 + nif;
	integer ids[MAX_FIELD_NUM + 1];
	dicts = new map<string, integer>[ndicts];
	memset(ids, 0, sizeof(integer)* (ndicts));
	for (i = 0; i < nif; ++i) dicts[i].insert(pair<string, integer>(NIL, ids[i]++));
	const integer MAX_WORD2VEC_LINE_LEN = 100000;
	char w2vline[MAX_WORD2VEC_LINE_LEN];
	long long nwords = 0;
	long long nvecds = 0;
	long long n = 0;
	char * pStart;
	fscanf(word2vec_fin, "%lld ", &nwords);
	fscanf(word2vec_fin, "%lld\n", &nvecds);
	if (nvecds != nfd[0]) {
		fprintf(stderr, "word2vec dim is not equal to nfd[0]\n");
		exit(1);
	}
	feature_table = (real **)malloc(nif * sizeof(real*));
	feature_table[0] = (real*)malloc((nwords + 1) *  nfd[0] * sizeof(real));
	for (n = 0; n < nvecds; ++n) feature_table[0][n] = 0;
	while (!feof(word2vec_fin)) {
		fgets(w2vline, MAX_WORD2VEC_LINE_LEN, word2vec_fin);
		i = strlen(w2vline) - 1;
		while (i >= 0 && (w2vline[i] == '\r' || w2vline[i] == '\n')) w2vline[i--] = '\0';
		if (i == -1) continue;
		pStart = w2vline;
		for (j = 0; pStart[j] != ' '; ++j) NULL;
		fld.assign(pStart, j);
		if (dicts[0].find(fld) == dicts[0].end()) dicts[0].insert(pair<string, integer>(fld, ids[0]++));
		else continue;
		pStart += j + 1;
		for (i = 0; i < nvecds; ++i) {
			for (j = 0; pStart[j] && pStart[j] != ' '; ++j) NULL;
			pStart[j] = '\0';
			feature_table[0][n++] = atof(pStart);
			pStart += j + 1;
		}
	}
	while (!feof(train_fin)){
		while (!feof(train_fin)) {
			ch = fgetc(train_fin);
			if (ch == '\n') ++train_line_num;
			else if (ch != '\r'){
				ungetc(ch, train_fin);
				break;
			}
		}
		if (feof(train_fin)) break;
		do{
			j = 0;
			for (i = 0; !feof(train_fin);){
				ch = fgetc(train_fin);
				if (ch == '\n') {
					++train_line_num;
					buffer[i] = 0;
					if (!buffer[0]) {
						fprintf(stderr, "Error while reading training file at %lld line. Unexpected new line is read.\n", train_line_num);
						exit(1);
					}
					fld.assign(buffer, i);
					if (dicts[nif].find(fld) == dicts[nif].end()) dicts[nif].insert(pair<string, integer>(fld, ids[nif]++));
					j = 1;
					break;
				}
				else if (ch == '\t'){
					buffer[i] = 0;
					break;
				}
				else if (ch != '\r'){
					if (i == MAX_WORD_LEN - 2){
						fprintf(stderr, "Error while reading training file at %lld line. There is a word with too long length.\n", train_line_num);
						exit(1);
					}
					buffer[i++] = ch;
				}
			}
			if (feof(train_fin)) {
				buffer[i] = 0;
				if (!buffer[0]) {
					fprintf(stderr, "Error while reading training file at %lld line. There is no label for the last example.\n", train_line_num);
					exit(1);
				}
				fld.assign(buffer, i);
				if (dicts[nif].find(fld) == dicts[nif].end()) dicts[nif].insert(pair<string, integer>(fld, ids[nif]++));
				j = 1;
			}
			if (j) break;
			for (nflds = 0, i = 0; !feof(train_fin);){
				ch = fgetc(train_fin);
				if (ch == '\n') {
					++train_line_num;
					if (nflds + 1 != nif){
						fprintf(stderr, "Error while reading training file at %lld line. Number of fields is not right.\n", train_line_num);
						exit(1);
					}
					buffer[i] = 0;
					fld.assign(buffer, i);
					if (nflds != 0 && dicts[nflds].find(fld) == dicts[nflds].end()) dicts[nflds].insert(pair<string, integer>(fld, ids[nflds]++));
					break;
				}
				else if (ch == '\t'){
					if (nflds + 1 >= nif){
						fprintf(stderr, "Error while reading training file at %lld line. Number of fields is not right.\n", train_line_num);
						exit(1);
					}
					buffer[i] = 0;
					fld.assign(buffer, i);
					if (nflds != 0 && dicts[nflds].find(fld) == dicts[nflds].end()) dicts[nflds].insert(pair<string, integer>(fld, ids[nflds]++));
					++nflds;
					i = 0;
				}
				else if (ch != '\r') {
					if (i == MAX_WORD_LEN - 2){
						fprintf(stderr, "Error while reading training file at %lld line. There is a word with too long length.\n", train_line_num);
						exit(1);
					}
					buffer[i++] = ch;
				}
			}
		} while (!feof(train_fin));
	}
	nor = dicts[nif].size();
	fprintf(stdout, "Finish making dictionary. there are %lld lines in train file.\n", train_line_num);
}

void SentenceClassification::get_sentence_classifier_example(FILE * fin, string * sen,	map<string, integer>::iterator **features,	map<string, integer>::iterator &label, integer &lst, long long &linu){
	char buffer[MAX_WORD_LEN], ch = 0;
	integer i = 0, j = 0, nflds = 0;
	string fld;
	lst = 0;
	if (feof(fin)) return;
	else while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == '\n') ++linu;
		else if (ch != '\r'){
			ungetc(ch, fin);
			break;
		}
	}
	if (feof(fin)) return;
	do{
		if (lst == MAX_SEN_LEN)	for (lst = 0, j = 1; !feof(fin);){
			ch = fgetc(fin);
			if (ch == '\n') {
				++linu;
				if (j) {
					while (!feof(fin)) {
						ch = fgetc(fin);
						if (ch == '\n') ++linu;
						else if (ch != '\r'){
							ungetc(ch, fin);
							break;
						}
					}
					break;
				}
				else j = 1;
			}
			else if (ch == '\t') j = 0;
		}
		if (feof(fin)) return;
		for (i = 0; !feof(fin);){
			ch = fgetc(fin);
			if (ch == '\n') {
				++linu;
				buffer[i] = 0;
				if (!buffer[0]) {
					fprintf(stderr, "\nError while reading file at %lld line. Unexpected new line is read.\n", linu);
					exit(1);
				}
				fld.assign(buffer, i);
				label = dicts[nif].find(fld);
				return;
			}
			else if (ch == '\t'){
				buffer[i] = 0;
				sen[lst].assign(buffer, i);
				break;
			}
			else if (ch != '\r'){
				if (i == MAX_WORD_LEN - 2){
					fprintf(stderr, "\nError while reading file at %lld line. There is a word with too long length.\n", linu);
					exit(1);
				}
				buffer[i++] = ch;
			}
		}
		if (feof(fin)) {
			buffer[i] = 0;
			if (!buffer[0]) {
				fprintf(stderr, "\nError while reading file at %lld line. There is no label for the last example.\n", linu);
				exit(1);
			}
			fld.assign(buffer, i);
			label = dicts[nif].find(fld);
			return;
		}
		for (nflds = 0, i = 0; !feof(fin);){
			ch = fgetc(fin);
			if (ch == '\n') {
				++linu;
				if (nflds + 1 != nif){
					fprintf(stderr, "\nError while reading file at %lld line. Number of fields is not right.\n", linu);
					exit(1);
				}
				buffer[i] = 0;
				fld.assign(buffer, i);
				if ((label = dicts[nflds].find(fld)) == dicts[nflds].end()) label = dicts[nflds].find(NIL);
				features[nflds][lst] = label;
				++lst;
				break;
			}
			else if (ch == '\t'){
				if (nflds + 1 >= nif){
					fprintf(stderr, "\nError while reading file at %lld line. Number of fields is not right.\n", linu);
					exit(1);
				}
				buffer[i] = 0;
				fld.assign(buffer, i);
				if ((label = dicts[nflds].find(fld)) == dicts[nflds].end()) label = dicts[nflds].find(NIL);
				features[nflds++][lst] = label;
				i = 0;
			}
			else if (ch != '\r') {
				if (i == MAX_WORD_LEN - 2){
					fprintf(stderr, "\nError while reading file at %lld line. There is a word with too long length.\n", linu);
					exit(1);
				}
				buffer[i++] = ch;
			}
		}
	} while (!feof(fin));
}

void SentenceClassification::load_model(){
	fprintf(stdout, "Start loading model\n");
	integer i1, i2, i3, i4, i5, tmp, csize, dictsize, index;
	char fld[MAX_WORD_LEN];
	fscanf(model_file, "nif = %d\n", &nif);
	nfd = (integer *)malloc(sizeof(integer)* nif);
	fscanf(model_file, "nfd = ");
	for (i1 = 0; i1 < nif; ++i1) fscanf(model_file, "%d ", &nfd[i1]);
	fscanf(model_file, "\nnwd = %d\n", &nwd);
	fscanf(model_file, "ncl = %d\n", &ncl);
	scw = (integer *)malloc(sizeof(integer)* ncl);
	ncr = (integer *)malloc(sizeof(integer)* ncl);
	ncm = (integer *)malloc(sizeof(integer)* ncl);
	fscanf(model_file, "scw = ");
	for (i1 = 0; i1 < ncl; ++i1) fscanf(model_file, "%d ", &scw[i1]);
	fscanf(model_file, "\nncr = ");
	for (i1 = 0; i1 < ncl; ++i1) fscanf(model_file, "%d ", &ncr[i1]);
	fscanf(model_file, "\nncm = ");
	for (i1 = 0; i1 < ncl; ++i1) fscanf(model_file, "%d ", &ncm[i1]);
	fscanf(model_file, "\nKtop = %d", &Ktop);
	fscanf(model_file, "\nnhl = %d", &nhl);
	nhr = (integer *)malloc(sizeof(integer)* nhl);
	fscanf(model_file, "\nnhr = ");
	for (i1 = 0; i1 < nhl; ++i1) fscanf(model_file, "%d ", &nhr[i1]);
	fscanf(model_file, "\n");
	fscanf(model_file, "\n");
	fprintf(stdout, "\tnif = %d\n", nif);
	fprintf(stdout, "\tnfd = ");
	for (i1 = 0; i1 < nif; ++i1) fprintf(stdout, "%d ", nfd[i1]);
	fprintf(stdout, "\n\tnwd = %d\n", nwd);
	fprintf(stdout, "\tncl = %d\n", ncl);
	fprintf(stdout, "\tscw = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(stdout, "%d ", scw[i1]);
	fprintf(stdout, "\n\tncr = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(stdout, "%d ", ncr[i1]);
	fprintf(stdout, "\n\tncm = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(stdout, "%d ", ncm[i1]);
	fprintf(stdout, "\n\tKtop = %d", Ktop);
	fprintf(stdout, "\n\tnhl = %d", nhl);
	fprintf(stdout, "\n\tnhr = ");
	for (i1 = 0; i1 < nhl; ++i1) fprintf(stdout, "%d ", nhr[i1]);
	fprintf(stdout, "\n");
	dicts = new map<string, integer>[nif + 1];
	for (i1 = 0, tmp = nif + 1; i1 < tmp; ++i1){
		fscanf(model_file, "dicts[%d]\tsize = %d\n", &i1, &dictsize);
		for (i2 = 0; i2 < dictsize; ++i2) {
			fscanf(model_file, "%s\t%d\n", fld, &index);
			dicts[i1].insert(pair<string, integer>(string(fld), index));
		}
		fscanf(model_file, "\n");
	}
	nor = static_cast<integer>(dicts[nif].size());
	alloc_mem();
	for (i1 = 0; i1 < nif; ++i1){
		fscanf(model_file, "feature_table[%d]\tsize = %d * %d\n", &index, &dictsize, &nfd[i1]);
		for (i2 = 0; i2 < dicts[i1].size(); ++i2){
			for (i3 = 0; i3 < nfd[i1]; ++i3) fscanf(model_file, "%lf ", &feature_table[i1][i2 * nfd[i1] + i3]);
			fscanf(model_file, "\n");
		}
		fscanf(model_file, "\n");
	}
	csize = nwd * scw[0] + 1;
	for (i2 = 0; i2 < ncm[0]; ++i2) {
		fscanf(model_file, "MM[0][%d][0]\t size = %d * %d\n", &index, &ncr[0], &csize);
		for (i4 = 0; i4 < ncr[0]; ++i4){
			for (i5 = 0; i5 < csize; ++i5)
				fscanf(model_file, "%lf ", &MM[0][i2][i4 * csize + i5]);
			fscanf(model_file, "\n");
		}
		fscanf(model_file, "\n");
	}
	for (i1 = 1, tmp = 0; i1 < ncl; ++i1, ++tmp){
		csize = ncr[tmp] * scw[i1] + 1;
		for (i2 = 0; i2 < ncm[i1]; ++i2) for (i3 = 0; i3 < ncm[tmp]; ++i3){
			fscanf(model_file, "MM[%d][%d][%d]\t size = %d * %d\n", &index, &index, &index, &ncr[i1], &csize);
			for (i4 = 0; i4 < ncr[i1]; ++i4){
				for (i5 = 0; i5 < csize; ++i5) fscanf(model_file, "%lf ", &MM[i1][i2 * ncm[tmp] + i3][i4 * csize + i5]);
				fscanf(model_file, "\n");
			}
			fscanf(model_file, "\n");
		}
	}
	csize = ncm[ncl - 1] * Ktop * ncr[ncl - 1] + 1;
	fscanf(model_file, "WM[0]\tsize = %d * %d\n", &nhr[0], &csize);
	for (i2 = 0; i2 < nhr[0]; ++i2){
		for (i3 = 0; i3 < csize; ++i3) fscanf(model_file, "%lf ", &WM[0][i2 * csize + i3]);
		fscanf(model_file, "\n");
	}
	fscanf(model_file, "\n");
	for (i1 = 1, tmp = 0; i1 < nhl; ++i1, ++tmp){
		csize = nhr[tmp] + 1;
		fscanf(model_file, "WM[%d]\t size = %d * %d\n", &index, &nhr[i1], &csize);
		for (i2 = 0; i2 < nhr[i1]; ++i2){
			for (i3 = 0; i3 < csize; ++i3) fscanf(model_file, "%lf ", &WM[i1][i2 * csize + i3]);
			fscanf(model_file, "\n");
		}
		fscanf(model_file, "\n");
	}
	csize = nhr[nhl - 1] + 1;
	fscanf(model_file, "U\tsize = %d * %d\n", &nor, &csize);
	for (i1 = 0; i1 < nor; ++i1){
		for (i2 = 0; i2 < csize; ++i2) fscanf(model_file, "%lf ", &U[i1 * csize + i2]);
		fscanf(model_file, "\n");
	}
	fscanf(model_file, "\n");
	fprintf(stdout, "Finish loading model\n");
}

void SentenceClassification::save_model(){
	fprintf(stdout, "Start saving model\n");
	integer i1, i2, i3, i4, i5, tmp, csize;
	fprintf(model_file, "nif = %d\n", nif);
	fprintf(model_file, "nfd = ");
	for (i1 = 0; i1 < nif; ++i1) fprintf(model_file, "%d ", nfd[i1]);
	fprintf(model_file, "\nnwd = %d\n", nwd);
	fprintf(model_file, "ncl = %d\n", ncl);
	fprintf(model_file, "scw = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(model_file, "%d ", scw[i1]);
	fprintf(model_file, "\nncr = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(model_file, "%d ", ncr[i1]);
	fprintf(model_file, "\nncm = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(model_file, "%d ", ncm[i1]);
	fprintf(model_file, "\nKtop = %d", Ktop);
	fprintf(model_file, "\nnhl = %d", nhl);
	fprintf(model_file, "\nnhr = ");
	for (i1 = 0; i1 < nhl; ++i1) fprintf(model_file, "%d ", nhr[i1]);
	fprintf(model_file, "\n");
	fprintf(model_file, "\n");

	for (i1 = 0, tmp = nif + 1; i1 < tmp; ++i1){
		fprintf(model_file, "dicts[%d]\tsize = %d\n", i1, dicts[i1].size());
		for (map<string, integer>::iterator it = dicts[i1].begin(); it != dicts[i1].end(); ++it) fprintf(model_file, "%s\t%d\n", it->first.c_str(), it->second);
		fprintf(model_file, "\n");
	}
	for (i1 = 0; i1 < nif; ++i1){
		fprintf(model_file, "feature_table[%d]\tsize = %d * %d\n", i1, dicts[i1].size(), nfd[i1]);
		for (i2 = 0; i2 < dicts[i1].size(); ++i2){
			for (i3 = 0; i3 < nfd[i1]; ++i3) fprintf(model_file, "%lf ", feature_table[i1][i2 * nfd[i1] + i3]);
			fprintf(model_file, "\n");
		}
		fprintf(model_file, "\n");
	}
	csize = nwd * scw[0] + 1;
	for (i2 = 0; i2 < ncm[0]; ++i2) {
		fprintf(model_file, "MM[0][%d][0]\t size = %d * %d\n", i2, ncr[0], csize);
		for (i4 = 0; i4 < ncr[0]; ++i4){
			for (i5 = 0; i5 < csize; ++i5)
				fprintf(model_file, "%lf ", MM[0][i2][i4 * csize + i5]);
			fprintf(model_file, "\n");
		}
		fprintf(model_file, "\n");
	}
	for (i1 = 1, tmp = 0; i1 < ncl; ++i1, ++tmp){
		csize = ncr[tmp] * scw[i1] + 1;
		for (i2 = 0; i2 < ncm[i1]; ++i2) for (i3 = 0; i3 < ncm[tmp]; ++i3){
			fprintf(model_file, "MM[%d][%d][%d]\t size = %d * %d\n", i1, i2, i3, ncr[i1], csize);
			for (i4 = 0; i4 < ncr[i1]; ++i4){
				for (i5 = 0; i5 < csize; ++i5) fprintf(model_file, "%lf ", MM[i1][i2 * ncm[tmp] + i3][i4 * csize + i5]);
				fprintf(model_file, "\n");
			}
			fprintf(model_file, "\n");
		}
	}
	csize = ncm[ncl - 1] * Ktop * ncr[ncl - 1] + 1;
	fprintf(model_file, "WM[0]\tsize = %d * %d\n", nhr[0], csize);
	for (i2 = 0; i2 < nhr[0]; ++i2){
		for (i3 = 0; i3 < csize; ++i3) fprintf(model_file, "%lf ", WM[0][i2 * csize + i3]);
		fprintf(model_file, "\n");
	}
	fprintf(model_file, "\n");
	for (i1 = 1, tmp = 0; i1 < nhl; ++i1, ++tmp){
		csize = nhr[tmp] + 1;
		fprintf(model_file, "WM[%d]\t size = %d * %d\n", i1, nhr[i1], csize);
		for (i2 = 0; i2 < nhr[i1]; ++i2){
			for (i3 = 0; i3 < csize; ++i3) fprintf(model_file, "%lf ", WM[i1][i2 * csize + i3]);
			fprintf(model_file, "\n");
		}
		fprintf(model_file, "\n");
	}
	csize = nhr[nhl - 1] + 1;
	fprintf(model_file, "U\tsize = %d * %d\n", nor, csize);
	for (i1 = 0; i1 < nor; ++i1){
		for (i2 = 0; i2 < csize; ++i2) fprintf(model_file, "%lf ", U[i1 * csize + i2]);
		fprintf(model_file, "\n");
	}
	fprintf(model_file, "\n");
	fprintf(stdout, "Finish saving model\n");
}

void SentenceClassification::save_model_binary(){
	fprintf(stdout, "Start saving model\n");
	integer i1, i2, i3, tmp, csize;
	fwrite(&nif, sizeof(integer), 1, model_file);
	fwrite(nfd, sizeof(integer), nif, model_file);
	fwrite(&nwd, sizeof(integer), 1, model_file);
	fwrite(&ncl, sizeof(integer), 1, model_file);
	fwrite(scw, sizeof(integer), ncl, model_file);
	fwrite(ncr, sizeof(integer), ncl, model_file);
	fwrite(ncm, sizeof(integer), ncl, model_file);
	fwrite(&Ktop, sizeof(integer), 1, model_file);
	fwrite(&nhl, sizeof(integer), 1, model_file);
	fwrite(nhr, sizeof(integer), nhl, model_file);
	for (i1 = 0, tmp = nif + 1; i1 < tmp; ++i1){
		csize = static_cast<integer>(dicts[i1].size());
		fwrite(&csize, sizeof(integer), 1, model_file);
		for (map<string, integer>::iterator it = dicts[i1].begin(); it != dicts[i1].end(); ++it) {
			fprintf(model_file, "%s\n", it->first.c_str());
			csize = it->second;
			fwrite(&csize, sizeof(integer), 1, model_file);
		}
	}
	for (i1 = 0; i1 < nif; ++i1) {
		csize = static_cast<integer>(dicts[i1].size()) * nfd[i1];
		csize = fwrite(feature_table[i1], sizeof(real), csize, model_file);
		
	}
	csize = nwd * scw[0] + 1;
	for (i2 = 0; i2 < ncm[0]; ++i2) fwrite(MM[0][i2], sizeof(real), ncr[0] * csize, model_file);
	for (i1 = 1, tmp = 0; i1 < ncl; ++i1, ++tmp){
		csize = ncr[tmp] * scw[i1] + 1;
		for (i2 = 0; i2 < ncm[i1]; ++i2) for (i3 = 0; i3 < ncm[tmp]; ++i3) fwrite(MM[i1][i2 * ncm[tmp] + i3], sizeof(real), ncr[i1] * csize, model_file);
	}
	fwrite(WM[0], sizeof(real), nhr[0] * (ncm[ncl - 1] * Ktop * ncr[ncl - 1] + 1), model_file);
	for (i1 = 1, tmp = 0; i1 < nhl; ++i1, ++tmp) fwrite(WM[i1], sizeof(real), nhr[i1] * (nhr[tmp] + 1), model_file);
	fwrite(U, sizeof(real), nor * (nhr[nhl - 1] + 1), model_file);
	fprintf(stdout, "Finish saving model\n");
}

void SentenceClassification::load_model_binary(){
	fprintf(stdout, "Start loading model\n");
	integer i1, i2, i3, len, tmp, csize, index;
	char fld[MAX_WORD_LEN + 2];
	fread(&nif, sizeof(integer), 1, model_file);
	nfd = (integer *)malloc(sizeof(integer)* nif);
	fread(nfd, sizeof(integer), nif, model_file);
	fread(&nwd, sizeof(integer), 1, model_file);
	fread(&ncl, sizeof(integer), 1, model_file);
	scw = (integer *)malloc(sizeof(integer)* ncl);
	ncr = (integer *)malloc(sizeof(integer)* ncl);
	ncm = (integer *)malloc(sizeof(integer)* ncl);
	fread(scw, sizeof(integer), ncl, model_file);
	fread(ncr, sizeof(integer), ncl, model_file);
	fread(ncm, sizeof(integer), ncl, model_file);
	fread(&Ktop, sizeof(integer), 1, model_file);
	fread(&nhl, sizeof(integer), 1, model_file);
	nhr = (integer *)malloc(sizeof(integer)* nhl);
	fread(nhr, sizeof(integer), nhl, model_file);

	fprintf(stdout, "\tnif = %d\n", nif);
	fprintf(stdout, "\tnfd = ");
	for (i1 = 0; i1 < nif; ++i1) fprintf(stdout, "%d ", nfd[i1]);
	fprintf(stdout, "\n\tnwd = %d\n", nwd);
	fprintf(stdout, "\tncl = %d\n", ncl);
	fprintf(stdout, "\tscw = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(stdout, "%d ", scw[i1]);
	fprintf(stdout, "\n\tncr = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(stdout, "%d ", ncr[i1]);
	fprintf(stdout, "\n\tncm = ");
	for (i1 = 0; i1 < ncl; ++i1) fprintf(stdout, "%d ", ncm[i1]);
	fprintf(stdout, "\n\tKtop = %d", Ktop);
	fprintf(stdout, "\n\tnhl = %d", nhl);
	fprintf(stdout, "\n\tnhr = ");
	for (i1 = 0; i1 < nhl; ++i1) fprintf(stdout, "%d ", nhr[i1]);
	fprintf(stdout, "\n");

	dicts = new map<string, integer>[nif + 1];
	for (i1 = 0, tmp = nif + 1; i1 < tmp; ++i1){
		fread(&csize, sizeof(integer), 1, model_file);
		for (i2 = 0; i2 < csize; ++i2){
			fgets(fld, MAX_WORD_LEN + 2, model_file);
			len = strlen(fld);
			while ((fld[len - 1] == '\n' || fld[len - 1] == '\r') && len > 0) fld[--len] = 0;
			fread(&index, sizeof(integer), 1, model_file);
			dicts[i1].insert(pair<string, integer>(string(fld), index));
		}
	}
	nor = static_cast<integer>(dicts[nif].size());
	alloc_mem();
	for (i1 = 0; i1 < nif; ++i1)
	{
		csize = static_cast<integer>(dicts[i1].size()) * nfd[i1];
		csize = fread(feature_table[i1], sizeof(real), csize, model_file);
	}
	csize = nwd * scw[0] + 1;
	for (i2 = 0; i2 < ncm[0]; ++i2) fread(MM[0][i2], sizeof(real), ncr[0] * csize, model_file);
	for (i1 = 1, tmp = 0; i1 < ncl; ++i1, ++tmp){
		csize = ncr[tmp] * scw[i1] + 1;
		for (i2 = 0; i2 < ncm[i1]; ++i2) for (i3 = 0; i3 < ncm[tmp]; ++i3) fread(MM[i1][i2 * ncm[tmp] + i3], sizeof(real), ncr[i1] * csize, model_file);
	}
	fread(WM[0], sizeof(real), nhr[0] * (ncm[ncl - 1] * Ktop * ncr[ncl - 1] + 1), model_file);
	for (i1 = 1, tmp = 0; i1 < nhl; ++i1, ++tmp) fread(WM[i1], sizeof(real), nhr[i1] * (nhr[tmp] + 1), model_file);
	fread(U, sizeof(real), nor * (nhr[nhl - 1] + 1), model_file);
	fprintf(stdout, "Finish loading model\n");
}

void SentenceClassification::train_sentence_classifier_thread(integer thrd_id){
	fprintf(stdout, "Start train thread with id = %d\n", thrd_id);
	integer i, j;
	real * SENM;			//sentence matrix, with dimention of nwd * MAX_SEN_LEN
	real * lsz = 0;			//error of sentence matrix
	integer lst = 0;			//lenght of sentence
	integer * ncc = 0;			//number of convolution result colmuns per convolution layer 
	integer * ndk = 0;			//number of dynamic k-max pooling results per layer, calucateed by max{Ktop, upper( (ncl - l) * lst / ncl ) } 
	real *** CM = 0;		//convolution result matrix per convolution layer, each convolution layer have ncm matrixs, with dimension of -CM[0]: dcr[0] * (scw[0] + MAX_SEN_LEN -1) -CM[i]:  dcr[i] * (scw[i] + max{Ktop, upper( (ncl - i) * MAX_SEN_LEN / ncl ) } -1)
	real *** lcz = 0;		//convolution error term matrix per convolution layer, each convolution layer have ncm matrixs, with dimension of -CM[0]: dcr[0] * (scw[0] + MAX_SEN_LEN -1) -CM[i]:  dcr[i] * (scw[i] + max{Ktop, upper( (ncl - i) * MAX_SEN_LEN / ncl ) } -1)
	integer *** kpv = 0;		//k-max pooling result vector per convolution layer, each convolution layer have ncm results, stores the column number of CM
	real ** hve = 0;		//hidden layer vector per hidden layer,with dimension of  hve[i]:nhl
	real ** lhz = 0;		//hidden laryer error term vector per hidden layer,with dimension of  hve[i]:nhl
	real * ove = 0;			//output vector of DNN
	real * loz = 0;			//output error term vector
	real * lche = 0;		//a cache to save error term
	real z = 0;
	//alloc menory
	SENM = (real *)malloc(nwd * MAX_SEN_LEN * sizeof(real));
	lsz = (real *)malloc(nwd * MAX_SEN_LEN * sizeof(real));
	ncc = (integer *)malloc(ncl * sizeof(integer));
	ndk = (integer *)malloc(ncl * sizeof(integer));
	CM = (real ***)malloc(ncl * sizeof(real**));
	lcz = (real ***)malloc(ncl * sizeof(real**));
	for (i = 0; i < ncl; ++i) {
		CM[i] = (real**)malloc(ncm[i] * sizeof(real*));
		lcz[i] = (real**)malloc(ncm[i] * sizeof(real*));
	}
	for (j = 0; j < ncm[0]; ++j){
		CM[0][j] = (real *)malloc(ncr[0] * (scw[0] + MAX_SEN_LEN - 1) * sizeof(real));
		lcz[0][j] = (real *)malloc(ncr[0] * (scw[0] + MAX_SEN_LEN - 1) * sizeof(real));
	}
	for (i = 1; i < ncl; ++i) for (j = 0; j < ncm[i]; ++j) {
		CM[i][j] = (real *)malloc(ncr[i] * (scw[i] + dynamic_k(MAX_SEN_LEN, i - 1) - 1) * sizeof(real));
		lcz[i][j] = (real *)malloc(ncr[i] * (scw[i] + dynamic_k(MAX_SEN_LEN, i - 1) - 1) * sizeof(real));
	}
	kpv = (integer ***)malloc(ncl * sizeof(integer**));
	for (i = 0; i < ncl; ++i) {
		kpv[i] = (integer **)malloc(ncm[i] * sizeof(integer *));
		for (j = 0; j < ncm[i]; ++j) kpv[i][j] = (integer *)malloc(dynamic_k(MAX_SEN_LEN, i) * sizeof(integer));
	}
	hve = (real **)malloc(nhl * sizeof(real *));
	lhz = (real **)malloc(nhl * sizeof(real *));
	for (i = 0; i < nhl; ++i) {
		hve[i] = (real*)malloc(nhr[i] * sizeof(real));
		lhz[i] = (real*)malloc(nhr[i] * sizeof(real));
	}
	ove = (real *)malloc(nor * sizeof(real));
	loz = (real *)malloc(nor * sizeof(real));
	lche = (real*)malloc(ncr[0] * (scw[0] * nwd + 1) * sizeof(real));
	map<string, integer>::iterator **features;
	map<string, integer>::iterator label;
	features = new map<string, integer>::iterator *[nif];
	for (i = 0; i < nif; ++i) features[i] = new map<string, integer>::iterator[MAX_SEN_LEN];
	string sen[MAX_SEN_LEN];
	while (true) {
		mtx_tfi.lock();
		get_sentence_classifier_example(train_fin, sen, features, label, lst, linu);
		if (lst == 0){
			if (citer < iter_num){
				++citer;
				rewind(train_fin);
				mtx_tfi.unlock();
				continue;
			}
			else {
				mtx_tfi.unlock();
				break;
			}
		}
		if (++exa_num % LOG_PER_EXA == 0) {
			fprintf(stdout, "%10lld training examples have been read. Average error term is %10lf.\n", exa_num, avgerr / LOG_PER_EXA);
			avgerr = 0;
		}
		mtx_tfi.unlock();
		if (label == dicts[nif].end()){
			fprintf(stderr, "\nError while reading training file at %lld line. Label is not in the dictionary.\n", linu);
			exit(1);
		}
		for (i = 0; i < ncl; ++i) ndk[i] = dynamic_k(lst, i);
		look_up_table(features, SENM, lst);
		forward(SENM, lst, CM, ncc, kpv, ndk, hve, ove);
		//calculate error item for ove
		for (i = 0, z = 0; i < nor; ++i) z += exp(ove[i]);
		for (i = 0; i < nor; ++i) {
			loz[i] = exp(ove[i]) / z;
			avgerr += abs(loz[i]);
		}
		avgerr -= abs(loz[label->second]);
		loz[label->second] = loz[label->second] - 1;
		avgerr += abs(loz[label->second]);
		backward(SENM, lsz, lst, CM, lcz, ncc, kpv, ndk, hve, lhz, loz, lche);
		update_feature_table_with_regularization(features, lsz, lst);		
	}
	//free mem
	free(SENM);
	free(lsz);
	free(ncc);
	free(ndk);
	for (i = 0; i < ncl; ++i) {
		for (j = 0; j < ncm[i]; ++j) {
			free(CM[i][j]);
			free(lcz[i][j]);
		}
		free(CM[i]);
		free(lcz[i]);
	}
	free(CM);
	free(lcz);
	for (i = 0; i < ncl; ++i) {
		for (j = 0; j < ncm[i]; ++j) free(kpv[i][j]);
		free(kpv[i]);
	}
	free(kpv);
	for (i = 0; i < nhl; ++i) {
		free(hve[i]);
		free(lhz[i]);
	}
	free(hve);
	free(lhz);
	free(ove);
	free(loz);
	free(lche);
	for (i = 0; i < nif; ++i) delete[] features[i];
	delete[] features;
	fprintf(stdout, "Finish train thread with id = %d\n", thrd_id);
}

void SentenceClassification::predict_sentence_classifier_thread(integer thrd_id){
	fprintf(stdout, "Start predict thread with id = %d\n", thrd_id);
	integer i, j;
	real * SENM;			//sentence matrix, with dimention of nwd * MAX_SEN_LEN
	integer lst = 0;			//lenght of sentence
	integer * ncc = 0;			//number of convolution result colmuns per convolution layer 
	integer * ndk = 0;			//number of dynamic k-max pooling results per layer, calucateed by max{Ktop, upper( (ncl - l) * lst / ncl ) } 
	real *** CM = 0;		//convolution result matrix per convolution layer, each convolution layer have ncm matrixs, with dimension of -CM[0]: dcr[0] * (scw[0] + MAX_SEN_LEN -1) -CM[i]:  dcr[i] * (scw[i] + max{Ktop, upper( (ncl - i) * MAX_SEN_LEN / ncl ) } -1)
	integer *** kpv = 0;		//k-max pooling result vector per convolution layer, each convolution layer have ncm results, stores the column number of CM
	real ** hve = 0;		//hidden layer vector per hidden layer,with dimension of  hve[i]:nhl
	real * ove = 0;			//output vector of DNN
	real * p = 0;			//probability
	integer  maxPIn = 0;		//probability
	real z = 0;
	//alloc menory
	SENM = (real *)malloc(nwd * MAX_SEN_LEN * sizeof(real));
	ncc = (integer *)malloc(ncl * sizeof(integer));
	ndk = (integer *)malloc(ncl * sizeof(integer));
	CM = (real ***)malloc(ncl * sizeof(real**));
	for (i = 0; i < ncl; ++i) {
		CM[i] = (real**)malloc(ncm[i] * sizeof(real*));
	}
	for (j = 0; j < ncm[0]; ++j) CM[0][j] = (real *)malloc(ncr[0] * (scw[0] + MAX_SEN_LEN - 1) * sizeof(real));
	for (i = 1; i < ncl; ++i) for (j = 0; j < ncm[i]; ++j) 	CM[i][j] = (real *)malloc(ncr[i] * (scw[i] + dynamic_k(MAX_SEN_LEN, i - 1) - 1) * sizeof(real));
	kpv = (integer ***)malloc(ncl * sizeof(integer**));
	for (i = 0; i < ncl; ++i) {
		kpv[i] = (integer **)malloc(ncm[i] * sizeof(integer *));
		for (j = 0; j < ncm[i]; ++j) kpv[i][j] = (integer *)malloc(dynamic_k(MAX_SEN_LEN, i) * sizeof(integer));
	}
	hve = (real **)malloc(nhl * sizeof(real *));
	for (i = 0; i < nhl; ++i) hve[i] = (real*)malloc(nhr[i] * sizeof(real));
	ove = (real *)malloc(nor * sizeof(real));
	p = (real *)malloc(nor * sizeof(real));
	map<string, integer>::iterator **features;
	map<string, integer>::iterator label;
	features = new map<string, integer>::iterator *[nif];
	string sen[MAX_SEN_LEN];
	map<string, integer>::iterator * i2tag;
	i2tag = new map<string, integer>::iterator[dicts[nif].size()];
	for (label = dicts[nif].begin(); label != dicts[nif].end(); ++label) i2tag[label->second] = label;
	for (i = 0; i < nif; ++i) features[i] = new map<string, integer>::iterator[MAX_SEN_LEN];
	while (true) {
		mtx_pfi.lock();
		get_sentence_classifier_example(pred_fin, sen, features, label, lst, linu);
		mtx_pfi.unlock();
		if (lst == 0) break;
		for (i = 0; i < ncl; ++i) ndk[i] = dynamic_k(lst, i);
		look_up_table(features, SENM, lst);
		forward(SENM, lst, CM, ncc, kpv, ndk, hve, ove);
		for (i = 0, z = 0; i < nor; ++i) z += exp(ove[i]);
		maxPIn = 0;
		for (i = 0; i < nor; ++i) {
			p[i] = exp(ove[i]) / z;
			if (p[i] > p[maxPIn]) maxPIn = i;
		}
		mtx_pfo.lock();
		++crl_vec[maxPIn];
		if (label != dicts[nif].end()) {
			++grl_vec[label->second];
			if (maxPIn == label->second) ++rgt_vec[maxPIn];
		}
		if (pred_fout){
			for (i = 0; i < lst; ++i){
				fprintf(pred_fout, "%s", sen[i].c_str());
				for (j = 0; j < nif; ++j) fprintf(pred_fout, "\t%s", features[j][i]->first.c_str());
				fprintf(pred_fout, "\n");
			}
			if (label == dicts[nif].end()) fprintf(pred_fout, "%s\tNIL\t%lf\n\n", i2tag[maxPIn]->first.c_str(), p[maxPIn]);
			else fprintf(pred_fout, "%s\t%s\t%lf\n\n", i2tag[maxPIn]->first.c_str(), label->first.c_str(), p[maxPIn]);
		}
		mtx_pfo.unlock();
	}
	//free mem
	free(SENM);
	free(ncc);
	free(ndk);
	for (i = 0; i < ncl; ++i) {
		for (j = 0; j < ncm[i]; ++j) free(CM[i][j]);
		free(CM[i]);
	}
	free(CM);
	for (i = 0; i < ncl; ++i) {
		for (j = 0; j < ncm[i]; ++j) free(kpv[i][j]);
		free(kpv[i]);
	}
	free(kpv);
	for (i = 0; i < nhl; ++i) free(hve[i]);
	free(hve);
	free(ove);
	free(p);
	for (i = 0; i < nif; ++i) delete[] features[i];
	delete[] features;
	fprintf(stdout, "Finish predict thread with id = %d\n", thrd_id);
}

void SentenceClassification::train(integer argc, char **argv){
	if (argc == 1) {
		fprintf(stdout, "Sentence classification toolkit v 0.1b\n\n");
		fprintf(stdout, "Train options:\n");
		fprintf(stdout, "Parameters for training:\n");
		fprintf(stdout, "\t-train <file>\n");
		fprintf(stdout, "\t\tUse text data from <file> to train the model\n");
		fprintf(stdout, "\t-config <file>\n");
		fprintf(stdout, "\t\tUse configuration from <file> to construct the model\n");
		fprintf(stdout, "\t-model <file>\n");
		fprintf(stdout, "\t\tUse <file> to save the model\n");
		fprintf(stdout, "\t-word2vec <file>\n");
		fprintf(stdout, "\t\tUse <file> to load the word2vec\n");
		fprintf(stdout, "\t-iter <integer>\n");
		fprintf(stdout, "\t\tUse <integer> iteration number (default 1000)\n");
		fprintf(stdout, "\t-threads <integer>\n");
		fprintf(stdout, "\t\tUse <integer> threads (default 1)\n");
		fprintf(stdout, "\t-alpha <float>\n");
		fprintf(stdout, "\t\tSet the learning rate <float>; (default is 0.01)\n");
		fprintf(stdout, "\t-lambda <float>\n");
		fprintf(stdout, "\t\tSet the regularization rate <float>; (default is 0, suguesting not larger than 1e-6)\n");
		fprintf(stdout, "\t-binary <integer>\n");
		fprintf(stdout, "\t\tSet save in binary mode; (default is 1)\n");
		fprintf(stdout, "For example:\n");
		fprintf(stdout, "./slc_train -train train.tsv -config config.txt -model model -iter 1000 -alpha 0.01 -lambda 0 -binary 0 -threads 1\n");
		exit(1);
	}
	integer i;
	if ((i = get_arg_pos((char *)"-train", argc, argv)) > 0) strcpy(train_file_name, argv[i + 1]);
	if ((i = get_arg_pos((char *)"-config", argc, argv)) > 0) strcpy(config_file_name, argv[i + 1]);
	if ((i = get_arg_pos((char *)"-model", argc, argv)) > 0) strcpy(model_file_name, argv[i + 1]);
	if ((i = get_arg_pos((char *)"-word2vec", argc, argv)) > 0) strcpy(word2vec_file_name, argv[i + 1]);
	if ((i = get_arg_pos((char *)"-iter", argc, argv)) > 0) iter_num = atoi(argv[i + 1]);
	if ((i = get_arg_pos((char *)"-threads", argc, argv)) > 0) thrd_num = atoi(argv[i + 1]);
	if ((i = get_arg_pos((char *)"-alpha", argc, argv)) > 0) alp = atof(argv[i + 1]);
	if ((i = get_arg_pos((char *)"-lambda", argc, argv)) > 0) lmd = atof(argv[i + 1]);
	if ((i = get_arg_pos((char *)"-binary", argc, argv)) > 0) bin = atoi(argv[i + 1]);
	if (train_file_name[0] == 0){
		fprintf(stderr, "Train file must be given in the paramters\n");
		exit(1);
	}
	if (config_file_name[0] == 0){
		fprintf(stderr, "Config file must be given in the paramters\n");
		exit(1);
	}
	if (model_file_name[0] == 0){
		fprintf(stderr, "Model file must be given in the paramters\n");
		exit(1);
	}
	train_fin = fopen(train_file_name, "r");
	if (!train_fin){
		fprintf(stderr, "Can not open train file for reading.\n");
		exit(1);
	}
	config_file = fopen(config_file_name, "r");
	if (!config_file){
		fprintf(stderr, "Can not open config file for reading.\n");
		exit(1);
	}
	if (bin) model_file = fopen(model_file_name, "wb");
	else model_file = fopen(model_file_name, "w");
	if (!model_file){
		fprintf(stderr, "Can not open model file for writing.\n");
		exit(1);
	}
	if (word2vec_file_name[0]) {
		word2vec_fin = fopen(word2vec_file_name, "r");
		if (!word2vec_fin){
			fprintf(stderr, "Can not open word2vec file for reading.\n");
			exit(1);
		}
	}
	fprintf(stdout, "Parameters:\n");
	fprintf(stdout, "\tTrain file name = %s\n", train_file_name);
	fprintf(stdout, "\tConfigurations file name = %s\n", config_file_name);
	fprintf(stdout, "\tModel file name = %s\n", model_file_name);
	fprintf(stdout, "\tWord2vec file name = %s\n", word2vec_file_name);
	fprintf(stdout, "\tIteration number = %d\n", iter_num);
	fprintf(stdout, "\tThread number = %d\n", thrd_num);
	fprintf(stdout, "\tLearning rate = %lf\n", alp);
	fprintf(stdout, "\tRegularization rate = %lf\n", lmd);
	read_config();
	fclose(config_file);
	check_config();
	if (word2vec_file_name[0]) {
		make_dict_using_word2vec();
		fclose(word2vec_fin);
	}
	else {
		make_dict();
	}
	alloc_mem();
	init();
	rewind(train_fin);
	vector<thread> threads;
	for (i = 0; i < thrd_num; ++i){
		threads.push_back(thread(std::mem_fn(&SentenceClassification::train_sentence_classifier_thread), this, i));
	}
	for (i = 0; i < thrd_num; ++i){
		threads.at(i).join();
	}
	fclose(train_fin);
	if (bin) save_model_binary();
	else save_model();
	fclose(model_file);
	free_men();
}

void SentenceClassification::predict(integer argc, char **argv){
	if (argc == 1) {
		fprintf(stdout, "Sentence classification toolkit v 0.1b\n\n");
		fprintf(stdout, "Predict options:\n");
		fprintf(stdout, "Parameters for training:\n");
		fprintf(stdout, "\t-predict <file>\n");
		fprintf(stdout, "\t\tUse text data from <file> to train the model\n");
		fprintf(stdout, "\t-model <file>\n");
		fprintf(stdout, "\t\tUse <file> to save the model\n");
		fprintf(stdout, "\t-out <file>\n");
		fprintf(stdout, "\t\tUse <file> to save the predict result\n");
		fprintf(stdout, "\t-threads <integer>\n");
		fprintf(stdout, "\t\tUse <integer> threads (default 1)\n");
		fprintf(stdout, "\t-binary <integer>\n");
		fprintf(stdout, "\t\tSet save in binary mode; (default is 1)\n");
		fprintf(stdout, "For example:\n");
		fprintf(stdout, "./slc_predict -predict test.tsv -model model -out out.tsv -binary 0 -threads 1\n");
		exit(1);
	}
	integer i;
	if ((i = get_arg_pos((char *)"-predict", argc, argv)) > 0) strcpy(predict_in_file_name, argv[i + 1]);
	if ((i = get_arg_pos((char *)"-model", argc, argv)) > 0) strcpy(model_file_name, argv[i + 1]);
	if ((i = get_arg_pos((char *)"-out", argc, argv)) > 0) strcpy(predict_out_file_name, argv[i + 1]);
	if ((i = get_arg_pos((char *)"-threads", argc, argv)) > 0) thrd_num = atoi(argv[i + 1]);
	if ((i = get_arg_pos((char *)"-binary", argc, argv)) > 0) bin = atoi(argv[i + 1]);
	if (predict_in_file_name[0] == 0){
		fprintf(stderr, "Predict file must be given in the paramters\n");
		exit(1);
	}
	if (model_file_name[0] == 0){
		fprintf(stderr, "Model file must be given in the paramters\n");
		exit(1);
	}
	pred_fin = fopen(predict_in_file_name, "r");
	if (!pred_fin){
		fprintf(stderr, "Can not open predict in file for reading.\n");
		exit(1);
	}
	if (bin) model_file = fopen(model_file_name, "rb");
	else model_file = fopen(model_file_name, "r");
	if (!model_file){
		fprintf(stderr, "Can not open model file for reading.\n");
		exit(1);
	}
	if (predict_out_file_name[0]){
		pred_fout = fopen(predict_out_file_name, "w");
		if (!pred_fout){
			fprintf(stderr, "Can not open predict out file for writing.\n");
			exit(1);
		}
	}
	fprintf(stdout, "Parameters:\n");
	fprintf(stdout, "\tPredict in file name = %s\n", predict_in_file_name);
	fprintf(stdout, "\tModel file name = %s\n", model_file_name);
	fprintf(stdout, "\tPredict out file name = %s\n", predict_out_file_name);
	fprintf(stdout, "\tThread number = %d\n", thrd_num);
	if (bin) load_model_binary();
	else load_model();
	fclose(model_file);
	rgt_vec = (integer *)calloc(nor, sizeof(integer));
	crl_vec = (integer *)calloc(nor, sizeof(integer));
	grl_vec = (integer *)calloc(nor, sizeof(integer));
	vector<thread> threads;
	for (i = 0; i < thrd_num; ++i){
		threads.push_back(thread(std::mem_fn(&SentenceClassification::predict_sentence_classifier_thread), this, i));
	}
	for (i = 0; i < thrd_num; ++i){
		threads.at(i).join();
	}
	//predict_sentence_classifier_thread(i);
	integer total = 0;
	integer pre = 0;
	real p, r, f;
	for (map<string, integer>::iterator it = dicts[nif].begin(); it != dicts[nif].end(); ++it){
		total += crl_vec[it->second];
		pre += rgt_vec[it->second];
		p = !crl_vec[it->second] ? 0 : (real)rgt_vec[it->second] / (real)crl_vec[it->second];
		r = !grl_vec[it->second] ? 0 : (real)rgt_vec[it->second] / (real)grl_vec[it->second];
		f = p + r == 0 ? 0.0 : 2.0 * p * r / (p + r);
		fprintf(stdout, "Label %s: pred = %-8d, given = %-8d, right = %-8d, prec = %-8.6lf, recall = %-8.6lf, f = %-8.6lf\n"
			, it->first.c_str(), crl_vec[it->second], grl_vec[it->second], rgt_vec[it->second]
			, p, r, f);
	}
	fprintf(stdout, "Total : all = %-8d, right = %-8d, prec = %-8.6lf\n", total, pre, !total ? 0 : (real)pre / total);
	free(rgt_vec);
	free(crl_vec);
	free(grl_vec);
	fclose(pred_fin);
	if (pred_fout) fclose(pred_fout);
	free_men();
}
