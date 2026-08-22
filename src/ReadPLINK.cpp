#include "ReadPLINK.h"
#include "Logger.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <mutex>

namespace fs = std::filesystem;


/**
 * @brief Detect companion files (.bim, .fam) for BED file
 *
 * @param bed_file Path to .bed file
 */
void Plink::detect_bed_companion_files(std::string const& bed_file)
{
    fs::path bed_path(bed_file);
    std::string stem = bed_path.stem().string();
    std::string dir = bed_path.parent_path().string();

    // Construct .bim and .fam paths
    pvar_path = (dir.empty() ? stem : dir + "/" + stem) + ".bim";
    psam_path = (dir.empty() ? stem : dir + "/" + stem) + ".fam";

    // Verify files exist
    if (!fs::exists(pvar_path)) {
        throw std::runtime_error("ERROR: Cannot find .bim file: " + pvar_path);
    }
    if (!fs::exists(psam_path)) {
        throw std::runtime_error("ERROR: Cannot find .fam file: " + psam_path);
    }

    spdlog::info("Detected companion files:");
    spdlog::info("  BIM: {}", pvar_path);
    spdlog::info("  FAM: {}", psam_path);
}

/**
 * @brief Detect companion files (.pvar, .psam) for PGEN file
 *
 * @param pgen_file Path to .pgen file
 */
void Plink::detect_pgen_companion_files(std::string const& pgen_file)
{
    fs::path pgen_path_obj(pgen_file);
    std::string stem = pgen_path_obj.stem().string();
    std::string dir = pgen_path_obj.parent_path().string();

    // Construct .pvar and .psam paths
    pvar_path = (dir.empty() ? stem : dir + "/" + stem) + ".pvar";
    psam_path = (dir.empty() ? stem : dir + "/" + stem) + ".psam";

    // Verify files exist
    if (!fs::exists(pvar_path)) {
        throw std::runtime_error("ERROR: Cannot find .pvar file: " + pvar_path);
    }
    if (!fs::exists(psam_path)) {
        throw std::runtime_error("ERROR: Cannot find .psam file: " + psam_path);
    }

    spdlog::info("Detected companion files:");
    spdlog::info("  PVAR: {}", pvar_path);
    spdlog::info("  PSAM: {}", psam_path);
}

/**
 * @brief Process PLINK header block - initialize file reader
 *
 * @param pgen_or_bed_file Path to .pgen or .bed file
 */
void Plink::process_plink_header_block(std::string const& pgen_or_bed_file)
{
    pgen_path = pgen_or_bed_file;

    // Check file extension to determine format
    fs::path file_path(pgen_or_bed_file);
    std::string ext = file_path.extension().string();

    if (ext == ".bed") {
        format_type = "BED";
        spdlog::info("Processing PLINK 1.x BED file: {}", pgen_or_bed_file);
        detect_bed_companion_files(pgen_or_bed_file);
    } else {
        throw std::runtime_error(
            "ERROR: Unsupported file extension '" + ext + "'. Only .bed is supported.");
    }

    // Read variant information from .bim
    read_bim_file(pvar_path);
    raw_variant_ct = static_cast<uint32_t>(variant_ids.size());

    // Validate BED magic bytes (0x6c 0x1b 0x01 = PLINK 1.x SNP-major format).
    // This requires no external library — plain C file I/O.
    {
        FILE* fp = std::fopen(pgen_or_bed_file.c_str(), "rb");
        if (!fp)
            throw std::runtime_error("ERROR: Cannot open BED file: " + pgen_or_bed_file);
        uint8_t magic[3] = {0, 0, 0};
        std::fread(magic, 1, 3, fp);
        std::fclose(fp);
        if (magic[0] != 0x6c || magic[1] != 0x1b || magic[2] != 0x01)
            throw std::runtime_error(
                "ERROR: Invalid BED file — bad magic bytes. "
                "File must be in PLINK 1.x SNP-major format.");
    }

    // Count samples from .fam (BED header does not store sample count)
    raw_sample_ct = 0;
    {
        std::ifstream fsam(psam_path);
        std::string tmpline;
        while (std::getline(fsam, tmpline)) {
            if (!tmpline.empty()) ++raw_sample_ct;
        }
    }
    if (raw_sample_ct == 0)
        throw std::runtime_error("ERROR: No samples found in " + psam_path);

    spdlog::info("****************************************************************************");
    spdlog::info("General information of PLINK file:");
    spdlog::info("  Format: BED");
    spdlog::info("  Number of variants: {}", raw_variant_ct);
    spdlog::info("  Number of samples: {}", raw_sample_ct);
    spdlog::info("****************************************************************************");

    if (raw_variant_ct == 0)
        throw std::runtime_error("ERROR: Number of variants in PLINK file is 0");
}

/**
 * @brief Read .fam file (PLINK 1.x format)
 *
 * .fam file format: FID IID PAT MAT SEX PHENO
 * We use FID_IID as sample identifier
 *
 * @param fam_file Path to .fam file
 * @return Vector of sample IDs
 */
std::ext::V_string Plink::read_fam_file(std::string const& fam_file)
{
    std::ext::V_string sample_ids;
    std::ifstream fin(fam_file);

    if (!fin) {
        throw std::runtime_error("ERROR: Cannot open .fam file: " + fam_file);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string fid, iid;
        iss >> fid >> iid;

        // Use IID only (consistent with BGEN sample file format)
        sample_ids.push_back(iid);
    }

    fin.close();

    if (sample_ids.size() != raw_sample_ct) {
        throw std::runtime_error(fmt::format(
            "ERROR: Number of samples in .fam file ({}) does not match "
            "PLINK file header ({})",
            sample_ids.size(), raw_sample_ct));
    }

    return sample_ids;
}

/**
 * @brief Read .psam file (PLINK 2.0 format)
 *
 * .psam file format: Header line followed by sample data
 * At minimum: #IID or IID column
 *
 * @param psam_file Path to .psam file
 * @return Vector of sample IDs
 */
std::ext::V_string Plink::read_psam_file(std::string const& psam_file)
{
    std::ext::V_string sample_ids;
    std::ifstream fin(psam_file);

    if (!fin) {
        throw std::runtime_error("ERROR: Cannot open .psam file: " + psam_file);
    }

    std::string line;
    bool found_header = false;
    int iid_col = -1;
    int fid_col = -1;

    // Read header to find IID column (and optionally FID)
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        // Skip comment lines, but check for column header
        if (line[0] == '#') {
            if (line.find("#IID") != std::string::npos || line.find("#FID") != std::string::npos) {
                found_header = true;
                std::istringstream iss(line);
                std::string col_name;
                int col_idx = 0;

                while (iss >> col_name) {
                    if (col_name == "#IID" || col_name == "IID") {
                        iid_col = col_idx;
                    } else if (col_name == "#FID" || col_name == "FID") {
                        fid_col = col_idx;
                    }
                    col_idx++;
                }
                break;
            }
            continue;
        }

        // If we reach data without finding header, assume first column is IID
        if (!found_header) {
            iid_col = 0;
            fin.seekg(0);  // Reset to beginning
            break;
        }
    }

    if (iid_col == -1) {
        throw std::runtime_error("ERROR: Cannot find IID column in .psam file");
    }

    // Read sample data
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        int col_idx = 0;
        std::string fid, iid;

        while (iss >> token) {
            if (col_idx == fid_col) {
                fid = token;
            } else if (col_idx == iid_col) {
                iid = token;
            }
            col_idx++;
        }

        // Combine FID and IID if both present
        std::string sample_id;
        if (!fid.empty() && fid_col != -1) {
            sample_id = fid + "_" + iid;
        } else {
            sample_id = iid;
        }
        sample_ids.push_back(sample_id);
    }

    fin.close();

    if (sample_ids.size() != raw_sample_ct) {
        throw std::runtime_error(fmt::format(
            "ERROR: Number of samples in .psam file ({}) does not match "
            "PLINK file header ({})",
            sample_ids.size(), raw_sample_ct));
    }

    return sample_ids;
}

/**
 * @brief Read .bim file (PLINK 1.x format)
 *
 * .bim file format: CHR SNP CM BP A1 A2
 *
 * @param bim_file Path to .bim file
 */
void Plink::read_bim_file(std::string const& bim_file)
{
    std::ifstream fin(bim_file);

    if (!fin) {
        throw std::runtime_error("ERROR: Cannot open .bim file: " + bim_file);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string chr, snp, cm, bp, a1, a2;
        iss >> chr >> snp >> cm >> bp >> a1 >> a2;

        chromosome.push_back(chr);
        variant_ids.push_back(snp);
        base_pair_pos.push_back(std::stoul(bp));
        allele_ref.push_back(a1);
        allele_alt.push_back(a2);
    }

    fin.close();
    spdlog::info("Read {} variants from .bim file", variant_ids.size());
}

/**
 * @brief Read .pvar file (PLINK 2.0 format)
 *
 * .pvar file format: Header followed by variant data
 * Columns: #CHROM POS ID REF ALT ...
 *
 * @param pvar_file Path to .pvar file
 */
void Plink::read_pvar_file(std::string const& pvar_file)
{
    std::ifstream fin(pvar_file);

    if (!fin) {
        throw std::runtime_error("ERROR: Cannot open .pvar file: " + pvar_file);
    }

    std::string line;
    bool found_header = false;
    int chrom_col = 0;
    int pos_col = 1;
    int id_col = 2;
    int ref_col = 3;
    int alt_col = 4;

    // Read header to find column positions
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        // Skip meta-information lines
        if (line.substr(0, 2) == "##") continue;

        // Column header line
        if (line[0] == '#') {
            found_header = true;
            std::istringstream iss(line);
            std::string col_name;
            int col_idx = 0;

            while (iss >> col_name) {
                if (col_name == "#CHROM" || col_name == "CHROM") {
                    chrom_col = col_idx;
                } else if (col_name == "POS") {
                    pos_col = col_idx;
                } else if (col_name == "ID") {
                    id_col = col_idx;
                } else if (col_name == "REF") {
                    ref_col = col_idx;
                } else if (col_name == "ALT") {
                    alt_col = col_idx;
                }
                col_idx++;
            }
            break;
        }
    }

    // Read variant data
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (iss >> token) {
            tokens.push_back(token);
        }

        if (tokens.size() < 5) {
            throw std::runtime_error("ERROR: Invalid .pvar file format - less than 5 columns");
        }

        chromosome.push_back(tokens[chrom_col]);
        variant_ids.push_back(tokens[id_col]);
        base_pair_pos.push_back(std::stoul(tokens[pos_col]));
        allele_ref.push_back(tokens[ref_col]);
        allele_alt.push_back(tokens[alt_col]);
    }

    fin.close();
    spdlog::info("Read {} variants from .pvar file", variant_ids.size());
}

/**
 * @brief Process PLINK sample block - read sample IDs and perform matching
 *
 * @param fam_or_psam_file Path to .fam/.psam file (can be empty if auto-detected)
 * @param use_fam_psam Whether to use provided file path
 * @param covmap Covariate data map
 * @param pheno_missing_key Missing value key
 * @param numSelCol Number of covariate columns
 * @param sam_size Number of samples in covariate file
 * @param id_path Path to ID file (optional)
 * @param match_ids Whether to perform ID matching
 */
void Plink::process_plink_sample_block(
    const char fam_or_psam_file[300],
    bool use_fam_psam,
    std::ext::UMap_str_VV_string covmap,
    std::string pheno_missing_key,
    int numSelCol,
    int sam_size,
    std::string id_path,
    bool match_ids)
{
    // Use provided file path or auto-detected path
    std::string sample_file;
    if (use_fam_psam && std::strlen(fam_or_psam_file) > 0) {
        sample_file = fam_or_psam_file;
    } else {
        sample_file = psam_path;  // Already set by detect_*_companion_files
    }

    // Read sample IDs based on format
    if (format_type == "BED") {
        sampleID_all = read_fam_file(sample_file);
    } else {
        sampleID_all = read_psam_file(sample_file);
    }

    spdlog::info("Read {} sample IDs from {}", sampleID_all.size(),
                format_type == "BED" ? ".fam" : ".psam");

    // If no ID matching required, use all samples
    if (!match_ids) {
        new_samSize = sampleID_all.size();
        sampleID = sampleID_all;
        plink_to_out.resize(sampleID_all.size());
        for (size_t i = 0; i < sampleID_all.size(); ++i) {
            plink_to_out[i] = i;
        }
        return;
    }

    // Perform ID matching with covariate file
    // Build map of covariate sample IDs
    std::unordered_map<std::string, int> cov_id_to_idx;
    int cov_idx = 0;
    for (const auto& entry : covmap) {
        cov_id_to_idx[entry.first] = cov_idx++;
    }

    // Create mapping from PLINK samples to output order
    plink_to_out.resize(raw_sample_ct, -1);  // -1 means excluded
    std::vector<std::string> matched_samples;
    int match_count = 0;

    for (uint32_t i = 0; i < raw_sample_ct; ++i) {
        const std::string& plink_id = sampleID_all[i];

        // Check if this sample exists in covariate file
        auto it = covmap.find(plink_id);
        if (it != covmap.end() && !it->second.empty()) {
            // Sample found and has non-missing covariates
            plink_to_out[i] = match_count;
            matched_samples.push_back(plink_id);
            match_count++;
        }
    }

    new_samSize = match_count;
    sampleID = matched_samples;

    spdlog::info("****************************************************************************");
    spdlog::info("ID matching results:");
    spdlog::info("  Samples in PLINK file: {}", raw_sample_ct);
    spdlog::info("  Samples in covariate file: {}", covmap.size());
    spdlog::info("  Matched samples: {}", new_samSize);
    spdlog::info("  Excluded samples: {}", raw_sample_ct - new_samSize);

    if (new_samSize > 0) {
        int print_n = std::min(5, new_samSize);
        spdlog::info("First {} matched sample IDs:", print_n);
        for (int i = 0; i < print_n; ++i) {
            spdlog::info("  {}", sampleID[i]);
        }
    }
    spdlog::info("****************************************************************************");

    if (new_samSize == 0) {
        throw std::runtime_error("ERROR: No samples matched between PLINK file and covariate file");
    }
}

/**
 * @brief Get variant positions and prepare for multi-threading
 *
 * @param threads Number of threads
 * @param includeVariantFile File with variants to include
 * @param do_filters Whether to filter variants
 */
void Plink::get_variant_positions(int threads, std::string includeVariantFile, bool do_filters)
{
    this->threads = threads;
    filterVariants = do_filters;

    // If filtering, read include list
    if (do_filters && !includeVariantFile.empty()) {
        std::ifstream fin(includeVariantFile);
        if (!fin) {
            throw std::runtime_error("ERROR: Cannot open variant include file: " + includeVariantFile);
        }

        std::unordered_set<std::string> include_set;
        std::string line;
        bool first_line = true;

        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            if (first_line) {
                first_line = false;
                continue;  // Skip header
            }
            include_set.insert(line);
        }
        fin.close();

        // Find indices of variants to include
        for (uint32_t i = 0; i < raw_variant_ct; ++i) {
            if (include_set.count(variant_ids[i])) {
                include_idx.push_back(i);
                includeVariantIndex.push_back(i);
            }
        }

        spdlog::info("Filtering: {} of {} variants selected", include_idx.size(), raw_variant_ct);

        if (include_idx.empty()) {
            throw std::runtime_error("ERROR: No variants from include file found in PLINK file");
        }
    } else {
        // Include all variants
        for (uint32_t i = 0; i < raw_variant_ct; ++i) {
            include_idx.push_back(i);
            includeVariantIndex.push_back(i);
        }
    }

    // Divide variants into blocks for multi-threading
    uint32_t n_variants = include_idx.size();
    uint32_t variants_per_thread = (n_variants + threads - 1) / threads;

    variant_begin.clear();
    variant_end.clear();
    keepVariants.clear();

    for (int t = 0; t < threads; ++t) {
        uint32_t start_idx = t * variants_per_thread;
        uint32_t end_idx = std::min((t + 1) * variants_per_thread, n_variants);

        if (start_idx >= n_variants) break;

        variant_begin.push_back(start_idx);
        variant_end.push_back(end_idx);

        // Store which variants this thread should process
        std::vector<uint32_t> thread_variants;
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            thread_variants.push_back(include_idx[i]);
        }
        keepVariants.push_back(thread_variants);
    }

    spdlog::info("Dividing PLINK file into {} block(s)", variant_begin.size());
    for (size_t i = 0; i < variant_begin.size(); ++i) {
        spdlog::info("  Block {}: variants {} to {} ({} variants)",
                    i, variant_begin[i], variant_end[i], variant_end[i] - variant_begin[i]);
    }
}

/**
 * @brief Stream genotype dosages from PLINK file
 *
 * Multi-threaded function to read and convert genotypes to dosages.
 * Dosage coding: 0 = homozygous ref, 1 = heterozygous, 2 = homozygous alt, -9 = missing
 *
 * @param plinkFile Path to PLINK file
 * @param plink Plink object reference
 * @param queue Output queue for streaming chunks
 * @param threads Number of threads
 * @param snps_per_chunk SNPs per chunk
 */
void calc_dosage_plink(std::string const& plinkFile,
                      Plink& plink,
                      BoundedChunkQueue& queue,
                      int threads,
                      int snps_per_chunk,
                      double maf)
{
    const double MAF = maf;
    const double maxMAF = 1 - MAF;
    // Partition variants into per-thread blocks
    plink.get_variant_positions(threads, plink.includeVariantFile, plink.filterVariants);

    const int sam_size = plink.new_samSize;
    const uint32_t raw_sample_ct = plink.raw_sample_ct;
    const int n_blocks = static_cast<int>(plink.variant_begin.size());

    spdlog::info("Starting PLINK genotype dosage calculation:");
    spdlog::info("  Output samples: {}, Variants: {}, Threads: {}, SNPs/chunk: {}",
                sam_size, plink.include_idx.size(), n_blocks, snps_per_chunk);

    // BED format: use standard C file I/O per-thread to avoid pgenlib's
    // concurrent PgrGet limitation. Each thread opens its own FILE* and reads
    // independently via fseek+fread — this is inherently thread-safe.
    // BED layout: 3-byte header, then ceil(N/4) bytes per variant.
    // 2-bit encoding per sample: 00=hom A1(alt)→2, 01=missing→-9, 10=het→1, 11=hom A2(ref)→0
    const uint64_t bytes_per_var = (raw_sample_ct + 3) / 4;
    static constexpr float BED_DOSAGE[4] = {2.0f, -9.0f, 1.0f, 0.0f};

    auto worker = [&](int t) {
        try {
            FILE* fp = std::fopen(plinkFile.c_str(), "rb");
            if (!fp) throw std::runtime_error("Cannot open BED file: " + plinkFile);

            std::vector<uint8_t> row_buf(bytes_per_var);

            uint32_t start_v = plink.variant_begin[t];
            uint32_t end_v   = plink.variant_end[t];

            for (uint32_t chunk_start = start_v; chunk_start < end_v;
                 chunk_start += static_cast<uint32_t>(snps_per_chunk))
            {
                uint32_t chunk_end  = std::min(
                    chunk_start + static_cast<uint32_t>(snps_per_chunk), end_v);
                int chunk_size = static_cast<int>(chunk_end - chunk_start);

                Chunk chunk_data;
                chunk_data.data = std::shared_ptr<float>(
                    new float[chunk_size * sam_size](), std::default_delete<float[]>());
                chunk_data.rsid.reserve(chunk_size);
                chunk_data.snpid.reserve(chunk_size);
                chunk_data.chr.reserve(chunk_size);
                chunk_data.pos.reserve(chunk_size);
                chunk_data.allele0.reserve(chunk_size);
                chunk_data.allele1.reserve(chunk_size);
                chunk_data.n_samples.reserve(chunk_size);
                chunk_data.af.reserve(chunk_size);
                chunk_data.gv.reserve(chunk_size);
                chunk_data.cols = sam_size;

                int kept = 0;
                for (int local_idx = 0; local_idx < chunk_size; ++local_idx) {
                    uint32_t variant_idx = plink.include_idx[chunk_start + local_idx];

                    // Seek and read this variant's raw bytes from BED file
                    uint64_t offset = 3ULL + (uint64_t)variant_idx * bytes_per_var;
                    if (std::fseek(fp, (long)offset, SEEK_SET) != 0)
                        throw std::runtime_error(fmt::format(
                            "fseek failed for variant {}", variant_idx));
                    if (std::fread(row_buf.data(), 1, bytes_per_var, fp) != bytes_per_var)
                        throw std::runtime_error(fmt::format(
                            "fread failed for variant {}", variant_idx));

                    // Write into the next output row; overwritten in place if this
                    // variant fails the MAF filter below.
                    float* out_row = chunk_data.data.get() + kept * sam_size;
                    for (int j = 0; j < sam_size; ++j) out_row[j] = -9.0f;

                    float sum = 0.0f, sq_sum = 0.0f;
                    int   n_valid = 0;

                    for (uint32_t plink_idx = 0; plink_idx < raw_sample_ct; ++plink_idx) {
                        int out_idx = plink.plink_to_out[plink_idx];
                        if (out_idx < 0) continue;

                        // BED: 4 samples per byte, 2 bits each (LSB first)
                        uint32_t geno = (row_buf[plink_idx >> 2] >> ((plink_idx & 3) << 1)) & 3;
                        float dosage = BED_DOSAGE[geno];

                        out_row[out_idx] = dosage;
                        if (dosage >= 0.0f) {
                            sum    += dosage;
                            sq_sum += dosage * dosage;
                            ++n_valid;
                        }
                    }

                    float af = 0.0f, gv = 0.0f;
                    if (n_valid > 0) {
                        float mean = sum / n_valid;
                        af = mean / 2.0f;
                        gv = sq_sum / n_valid - mean * mean;
                    }

                    if (af < MAF || af > maxMAF) continue;

                    chunk_data.rsid.push_back(plink.variant_ids[variant_idx]);
                    chunk_data.snpid.push_back(plink.variant_ids[variant_idx]);
                    chunk_data.chr.push_back(plink.chromosome[variant_idx]);
                    chunk_data.pos.push_back(std::to_string(plink.base_pair_pos[variant_idx]));
                    chunk_data.allele0.push_back(plink.allele_alt[variant_idx]);
                    chunk_data.allele1.push_back(plink.allele_ref[variant_idx]);
                    chunk_data.n_samples.push_back(std::to_string(n_valid));
                    chunk_data.af.push_back(af);
                    chunk_data.gv.push_back(gv);
                    ++kept;
                }

                if (kept == 0) continue;

                chunk_data.rows = kept;
                queue.push(std::move(chunk_data));
            }

            std::fclose(fp);

        } catch (const std::exception& e) {
            spdlog::error("Thread {} error: {}", t, e.what());
            throw;
        }
    };

    std::vector<std::thread> thread_vec;
    for (int t = 0; t < n_blocks; ++t)
        thread_vec.emplace_back(worker, t);

    for (auto& th : thread_vec)
        if (th.joinable()) th.join();

    queue.close();
    spdlog::info("PLINK genotype dosage calculation completed");
}
