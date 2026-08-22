#pragma once
#include <cstdint>
#include <cstdio>
#include "ReadFiles.h"
#include "BoundedQueue.h"

/**
 * @brief Class for reading PLINK 1.x BED/BIM/FAM genotype files.
 *
 * All genotype reading uses standard C file I/O (fopen/fseek/fread) with no
 * dependency on pgenlib or any LGPL/GPL library.  Each reader thread opens its
 * own independent FILE* handle so concurrent reads are inherently thread-safe.
 *
 * BED encoding (2 bits per sample, 4 samples per byte, LSB-first):
 *   00 = hom A1 (alt)  → dosage 2
 *   01 = missing       → dosage −9
 *   10 = het           → dosage 1
 *   11 = hom A2 (ref)  → dosage 0
 */
class Plink
{
    public:
        // File format type: currently only "BED" is supported
        std::string format_type;

        // File paths
        std::string pgen_path;      // .bed file path
        std::string pvar_path;      // .bim file path
        std::string psam_path;      // .fam file path

        // Sample information (from .fam)
        uint32_t raw_sample_ct = 0;   // Total samples in file
        int new_samSize = 0;           // Samples after filtering/matching
        std::ext::V_string sampleID;      // Matched sample IDs (in output order)
        std::ext::V_string sampleID_all;  // All sample IDs before matching
        std::ext::V_int plink_to_out;     // Mapping: PLINK sample idx -> output idx (-1 if excluded)
        std::ext::V_double new_covdata;   // Updated covariate data (for collinearity check)

        // Variant information (from .bim)
        uint32_t raw_variant_ct = 0;       // Total variants in file
        std::vector<std::string> variant_ids;
        std::vector<std::string> chromosome;
        std::vector<uint32_t>   base_pair_pos;
        std::vector<std::string> allele_ref;
        std::vector<std::string> allele_alt;

        // Variant filtering
        bool filterVariants = false;
        std::string includeVariantFile;
        std::vector<long int>    include_idx;
        std::vector<uint32_t>    includeVariantIndex;
        std::vector<std::vector<uint32_t>> keepVariants;

        // Covariate collinearity checking
        int numIntSelCol_new = 0;
        int numExpSelCol_new = 0;
        int numSelCol_new    = 0;
        std::ext::V_int excludeCol;

        // Multi-threading: variant blocks
        uint32_t threads = 1;
        std::vector<uint32_t> variant_begin;
        std::vector<uint32_t> variant_end;

        int phenoType = 0;

        // ----------------------------------------------------------------
        // Public interface
        // ----------------------------------------------------------------

        /** Read BED file header (validates magic bytes) and BIM/FAM metadata. */
        void process_plink_header_block(std::string const& bed_file);

        /** Read FAM sample IDs and match with covariate file. */
        void process_plink_sample_block(const char fam_or_psam_file[300],
                                       bool use_fam_psam,
                                       std::ext::UMap_str_VV_string covmap,
                                       std::string pheno_missing_key,
                                       int numSelCol,
                                       int sam_size,
                                       std::string id_path = "",
                                       bool match_ids = false);

        /** Partition variants into per-thread blocks, applying optional include-list filter. */
        void get_variant_positions(int threads,
                                  std::string includeVariantFile,
                                  bool do_filters);

        ~Plink() = default;

    private:
        std::ext::V_string read_fam_file(std::string const& fam_file);
        std::ext::V_string read_psam_file(std::string const& psam_file);
        void read_bim_file(std::string const& bim_file);
        void read_pvar_file(std::string const& pvar_file);
        void detect_bed_companion_files(std::string const& bed_file);
        void detect_pgen_companion_files(std::string const& pgen_file);
};

/**
 * @brief Stream BED genotype dosages in chunks to a consumer queue.
 *
 * Multi-threaded: each thread owns an independent FILE* and reads a disjoint
 * block of variants via fseek/fread.  Output Chunk format is identical to the
 * BGEN reader so the downstream GWAS pipeline requires no changes.
 */
void calc_dosage_plink(std::string const& plinkFile,
                      Plink& plink,
                      BoundedChunkQueue& queue,
                      int threads,
                      int snps_per_chunk,
                      double maf = 0.001);
