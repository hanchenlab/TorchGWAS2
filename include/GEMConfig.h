#pragma once
#include "ReadFiles.h"
/**
 * @brief Struct to hold the result of reading covariate data.
 * 
 */
struct GEMOptions 
{
    std::string pheno_add;
    std::string cov_add;
    char pheno_delim = ',';
    char cov_delim = ',';
    std::string geno_add;
    std::string sample_add = "";
    bool do_filters = false;
    bool use_sample_file = false;
    std::string includeVariantFile = "";
    double maf = 0.001;
    int stream_snps = 1;
    std::string sampleid_header_name;
    std::string random_slope_header_name = "";
    std::ext::V_string covariates;
    std::ext::V_string exposures;
    std::ext::V_string interactions;
    std::string missing_key = "NA";
    std::string kin_add;
    char kin_delim = ',';
    double kin_diag = 1;
    int threads;
    int num_chunks = 0;
    std::string corr_file = "correction.txt";
    std::string out_file = "output.txt";
    std::string log_file = "log.log";
    std::string null_log_file = "null_log.log";
    bool verbose = false;
    GEMOptions(); 
};
