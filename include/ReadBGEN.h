#pragma once
#include "../thirdparty/zstd-1.5.5/lib/zstd.h"
#include "../thirdparty/libdeflate-1.18/libdeflate.h"
#include "zlib.h"
#include "ReadFiles.h"
#include "BoundedQueue.h"

using uchar = unsigned char;
class Bgen 
{
    public:
        // For file
        FILE* fin;

        // For BGEN offset
        uint32_t offset;

        // For BGEN header block
        uint32_t Mbgen;
        uint32_t Nbgen;
        uint32_t CompressedSNPBlocks;
        uint32_t Layout;

        // For BGEN header-flag block;
        uint32_t SampleIdentifiers;
        // For ID matching
        int new_samSize;
        std::ext::V_string sampleID;
        //AllsampleIDs before matching
        std::ext::V_string sampleID_all;
        std::ext::V_int bgen_to_out;
        std::ext::V_double  new_covdata;//used for update collinear covariates
        std::vector<long int> include_idx;
        std::vector <long int> variant_pos;
        std::vector<unsigned int> includeVariantIndex;
        // For check of co-linear relations between covX;
        int numIntSelCol_new;
        int numExpSelCol_new;
        int numSelCol_new;
        std::ext::V_int excludeCol;
        // For multithreading BGEN file
        int phenoType;
        uint32_t threads;
        bool filterVariants;
        std::vector<uint32_t> Mbgen_begin;
        std::vector<uint32_t> Mbgen_end;
        std::vector<long long unsigned int> bgenVariantPos;
        std::vector<std::vector<uint32_t> > keepVariants;

        void process_bgen_header_block(std::string bgenfile);
        void process_bgen_sample_block(const char sample_file[300], bool use_sample, std::ext::UMap_str_VV_string covmap, std::string pheno_missing_key, int numSelCol, int sam_size, std::string id_path = "", bool match_ids = false);
        void get_position_bgen_variant(int threads, std::string includeVariantFile, bool do_filters);
};

// void gemBGEN(int thread_num, double sigma2, double* resid, double* XinvXTX, vector<double> miu, BinE binE, Bgen bgen, CommandLine cmd);
void bgen13_get_two_vals(const unsigned char* prob_start, uint32_t bit_precision, uintptr_t offset, uintptr_t* first_val_ptr, uintptr_t* second_val_ptr);

void calc_dosage(std::string const& bgenFile, Bgen &bgen, BoundedChunkQueue& queue, int threads, int snps_per_chunk, double maf = 0.001);
// void calc_dosage(const std::string& bgenFile, Bgen &bgen, BoundedChunkQueue& queue, int snps_per_chunk);

void Read_bgen_file(std::string const& bgenFile, Bgen bgen, int thread_num, int stream_snps, std::string outFile);




 



