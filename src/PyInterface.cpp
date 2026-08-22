#include "RunPipeline.h"
// #include "ParallelFileReader.h"
#include "BoundedQueue.h"
#include <pybind11/numpy.h>
#include <memory>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     


namespace py = pybind11;
// Lightweight Python-visible wrapper around the internal queue
struct DosageStream 
{
    std::shared_ptr<BoundedChunkQueue> q;
    explicit DosageStream(std::shared_ptr<BoundedChunkQueue> queue) : q(std::move(queue)) {}
};

// Forward declaration of helper used below
namespace 
{
    inline py::array_t<float> chunk_to_numpy(const Chunk& c);
}


/**
 * @brief Construct a new py module object
 * 
 */
//Module to be called in python
PYBIND11_MODULE(Mygen, m) 
{
    m.doc() = R"doc(
        GEM interface for BGEN dosage calculation.
        Note:
        Please use "" or [] if you do not want to pass a value for a specific argument.
        )doc";
    py::class_<GEMOptions>(m, "GEMOptions")
        .def(py::init<>())
        .def_readwrite("pheno_add", &GEMOptions::pheno_add)
        .def_readwrite("cov_add", &GEMOptions::cov_add)
        .def_readwrite("pheno_delim", &GEMOptions::pheno_delim)
        .def_readwrite("cov_delim", &GEMOptions::cov_delim)
        .def_readwrite("geno_add", &GEMOptions::geno_add)
        .def_readwrite("sample_add", &GEMOptions::sample_add)
        .def_readwrite("do_filters", &GEMOptions::do_filters)
        .def_readwrite("use_sample_file",  &GEMOptions::use_sample_file)
        .def_readwrite("includeVariantFile", &GEMOptions::includeVariantFile)
        .def_readwrite("maf", &GEMOptions::maf)
        .def_readwrite("stream_snps",  &GEMOptions::stream_snps)
        .def_readwrite("sampleid_header_name", &GEMOptions::sampleid_header_name)
        .def_readwrite("random_slope_header_name", &GEMOptions::random_slope_header_name)
        .def_readwrite("covariates", &GEMOptions::covariates)
        .def_readwrite("missing_key", &GEMOptions::missing_key)
        .def_readwrite("kin_add", &GEMOptions::kin_add)
        .def_readwrite("kin_delim", &GEMOptions::kin_delim)
        .def_readwrite("kin_diag", &GEMOptions::kin_diag)
        .def_readwrite("threads", &GEMOptions::threads)
        .def_readwrite("corr_file", &GEMOptions::corr_file)
        .def_readwrite("out_file", &GEMOptions::out_file)
        .def_readwrite("log_file", &GEMOptions::log_file)
        .def_readwrite("null_log_file", &GEMOptions::null_log_file)
        .def_readwrite("verbose", &GEMOptions::verbose);
        
        // Bind GEMRunner
        py::class_<GEMRunner>(m, "GEMRunner")
            // .def(py::init<const GEMOptions&>())  // constructor
            .def(py::init<GEMOptions const&, bool>(),
            py::arg("opt"), py::arg("match_ids") = false)
            .def_readonly("opt", &GEMRunner::opt)
            // .def("run_fit_nullmodel", &GEMRunner::run_fit_nullmodel)
            .def("run_fit_nullmodel",
                [](GEMRunner &self) {
                    try {
                        return self.run_fit_nullmodel();
                    } catch (const std::exception &e) {
                        // convert C++ std::exception → Python ValueError
                        throw py::value_error(e.what());
                    }
                }
            )
         
            .def("start_dosage_stream",
                [](GEMRunner& self, std::size_t queue_capacity, int snps_per_chunk){
                    auto q = std::make_shared<BoundedChunkQueue>(queue_capacity);
                    auto geno_file = self.opt.geno_add;
                    auto threads = self.opt.threads;
                    auto maf = self.opt.maf;
                    auto genofile_type = self.genofile_type;

                    // Branch based on genotype file format
                    if (genofile_type == "BGEN") {
                        // BGEN format - use existing calc_dosage
                        auto bgen_copy = self.bgen; // shallow copy; calc_dosage opens its own FILE handles
                        std::thread([q, bgen_copy, geno_file, threads, snps_per_chunk, maf]() mutable {
                            calc_dosage(geno_file, bgen_copy, *q, threads, snps_per_chunk, maf);
                        }).detach();
                    } else if (genofile_type == "BED" || genofile_type == "PGEN") {
                        // PLINK format - use calc_dosage_plink
                        auto plink_sptr = self.plink_sptr;
                        std::thread([q, plink_sptr, geno_file, threads, snps_per_chunk, maf]() mutable {
                            calc_dosage_plink(geno_file, *plink_sptr, *q, threads, snps_per_chunk, maf);
                        }).detach();
                    } else {
                        throw std::runtime_error("Unsupported genotype file format: " + genofile_type);
                    }

                    return DosageStream(q);
                },
                py::arg("queue_capacity"), py::arg("snps_per_chunk") = 1
            )
            .def("start_dosage_stream",
                [](GEMRunner& self, std::size_t queue_capacity){
                    int snps_per_chunk = self.opt.stream_snps > 0 ? self.opt.stream_snps : 1000;
                    auto q = std::make_shared<BoundedChunkQueue>(queue_capacity);
                    auto geno_file = self.opt.geno_add;
                    auto threads = self.opt.threads;
                    auto maf = self.opt.maf;
                    auto genofile_type = self.genofile_type;

                    // Branch based on genotype file format
                    if (genofile_type == "BGEN") {
                        // BGEN format - use existing calc_dosage
                        auto bgen_copy = self.bgen;
                        std::thread([q, bgen_copy, geno_file, threads, snps_per_chunk, maf]() mutable {
                            calc_dosage(geno_file, bgen_copy, *q, threads, snps_per_chunk, maf);
                        }).detach();
                    } else if (genofile_type == "BED" || genofile_type == "PGEN") {
                        // PLINK format - use calc_dosage_plink
                        auto plink_sptr = self.plink_sptr;
                        std::thread([q, plink_sptr, geno_file, threads, snps_per_chunk, maf]() mutable {
                            calc_dosage_plink(geno_file, *plink_sptr, *q, threads, snps_per_chunk, maf);
                        }).detach();
                    } else {
                        throw std::runtime_error("Unsupported genotype file format: " + genofile_type);
                    }

                    return DosageStream(q);
                },
                py::arg("queue_capacity")
            );

    // Python-visible stream wrapper; keeps queue internal.
    py::class_<DosageStream>(m, "DosageStream")
        .def("close", [](DosageStream& s){ if (s.q) s.q->close(); })
        .def("__del__", [](DosageStream& s){ if (s.q) s.q->close(); })
        .def("size", [](DosageStream& s) -> std::size_t {
            if (!s.q) return 0;
            return s.q->size();
        })
        .def("capacity", [](DosageStream& s) -> std::size_t {
            if (!s.q) return 0;
            return s.q->capacity();
        })
        .def("__iter__", [](DosageStream& self) -> DosageStream& { return self; }, py::return_value_policy::reference_internal)
        .def("__next__", [](DosageStream& s) -> py::tuple {
            if (!s.q) throw py::stop_iteration();
            Chunk c;
            if (!s.q->pop(c)) {
                throw py::stop_iteration();
            }
            // --- Convert dosage matrix to NumPy array ---
        py::array_t<float> dosage = chunk_to_numpy(c);

        // --- Convert metadata vectors to Python dict ---
        py::dict meta;
        meta["SNPID"]          = c.snpid;
        meta["RSID"]           = c.rsid;
        meta["CHR"]            = c.chr;
        meta["POS"]            = c.pos;
        meta["Non_Effect_Allele"] = c.allele1;
        meta["Effect_Allele"]  = c.allele0;
        meta["N_Samples"]      = c.n_samples;
        meta["AF"]             = c.af;
        meta["GV"]             = c.gv;

        // return dosage;
        return py::make_tuple(dosage, meta);
        });
}
// Local helper to convert a Chunk to a zero-copy NumPy array with correct lifetime
namespace 
{
    inline py::array_t<float> chunk_to_numpy(const Chunk& c) 
    {
        float* ptr = c.data.get();
        // Keep the buffer alive by attaching a shared_ptr<float> into a capsule
        auto owner = new std::shared_ptr<float>(c.data);
        auto base = py::capsule(owner, [](void* p){ delete reinterpret_cast<std::shared_ptr<float>*>(p); });
        // Construct typed NumPy array with shape, strides, data pointer, and base capsule
        return py::array_t<float>(
            { static_cast<py::ssize_t>(c.rows), static_cast<py::ssize_t>(c.cols) },
            { static_cast<py::ssize_t>(c.cols * sizeof(float)), static_cast<py::ssize_t>(sizeof(float)) },
            ptr,
            base
        );
    }
}