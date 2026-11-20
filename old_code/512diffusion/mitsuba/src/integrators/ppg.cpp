#include "MPG/dtree.h"

#include <enoki/stl.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-ppg:

Path guiding baseline (:monosp:`ppg`)
-------------------------------------------

This plugin provides a standalone surface path guiding baseline, decoupled from the main MPG pipeline;
it only reuses directional distribution scaffolding so paper-style guiding comparisons can live in-repo.

 */

template <typename Float, typename Spectrum>
class PPGIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    using Scalar = scalar_t<Float>;
    using DTreeRecordType = DTreeRecord<Scalar, Spectrum>;
    using DTreeWrapperType = DTreeWrapper<Scalar, Spectrum>;

    struct SpatialCell {
        mutable std::mutex mutex;
        std::vector<DTreeRecordType> records;
        std::shared_ptr<DTreeWrapperType> tree;
        size_t total_records = 0;
        size_t last_rebuild_records = 0;
        bool rebuild_in_progress = false;
    };

    PPGIntegrator(const Properties &props) : Base(props) {
        m_train_auto = props.bool_("train_auto", true);
        m_train_budget = clamp(props.float_("train_budget", 1.0f), Scalar(0.f), Scalar(1.f));
        m_guide_fraction = clamp(props.float_("guide_fraction", 0.5f), Scalar(0.f), Scalar(1.f));
        m_samples_per_pass = props.int_("samples_per_pass", 1);
        m_spatial_resolution = std::max(1, props.int_("spatial_resolution", 8));
        m_min_tree_records = (size_t) std::max(64, props.int_("min_tree_records", 128));
        m_rebuild_interval = (size_t) std::max(16, props.int_("rebuild_interval", 64));
        m_max_records_per_cell = (size_t) std::max(256, props.int_("max_records_per_cell", 4096));
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        ensure_grid(scene);

        RayDifferential3f ray = ray_;
        Float eta(1.f);
        Float emission_weight(1.f);
        Spectrum throughput(1.f), result(0.f);

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        si.compute_partials(ray);
        Mask valid_ray = si.is_valid();
        EmitterPtr emitter = si.emitter(scene);

        bool specular_camera_path = true;

        for (int depth = 1;; ++depth) {
            if (any_or<true>(neq(emitter, nullptr))) {
                if (emitter->is_environment() || !specular_camera_path) {
                    result[active] += emission_weight * throughput * emitter->eval(si, active);
                }
            }

            active &= si.is_valid();

            if (depth > m_rr_depth) {
                Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                active &= sampler->next_1d(active) < q;
                throughput *= rcp(q);
            }

            if ((uint32_t) depth >= (uint32_t) m_max_depth ||
                ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active))) {
                break;
            }

            BSDFContext ctx;
            ctx.sampler = sampler;
            BSDFPtr bsdf = si.bsdf(ray);
            Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            if (likely(any_or<true>(active_e))) {
                result[active_e] += sample_direct_guided(scene, sampler, si, bsdf, ctx, throughput, active_e);
            }

            auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active),
                                               sampler->next_2d(active), active);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            if (!has_flag(bs.sampled_type, BSDFFlags::Delta)) {
                specular_camera_path = false;
            }

            throughput = throughput * bsdf_val;
            active &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active)) {
                break;
            }

            eta *= bs.eta;

            ray = si.spawn_ray(si.to_world(bs.wo));
            SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

            emitter = si_bsdf.emitter(scene, active);
            DirectionSample3f ds(si_bsdf, si);
            ds.object = emitter;

            if (any_or<true>(neq(emitter, nullptr))) {
                Float emitter_pdf =
                    select(neq(emitter, nullptr) && !has_flag(bs.sampled_type, BSDFFlags::Delta),
                           scene->pdf_emitter_direction(si, ds), 0.f);
                Float direct_pdf = mixed_direct_pdf(si, ds.d, emitter_pdf);
                emission_weight = mis_weight(bs.pdf, direct_pdf);
            } else {
                emission_weight = 1.f;
            }

            si = std::move(si_bsdf);
        }

        return { result, valid_ray };
    }

    std::string to_string() const override {
        return tfm::format(
            "PPGIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i,\n"
            "  train_auto = %s,\n"
            "  train_budget = %f,\n"
            "  guide_fraction = %f,\n"
            "  spatial_resolution = %i,\n"
            "  min_tree_records = %zu,\n"
            "  rebuild_interval = %zu,\n"
            "  max_records_per_cell = %zu\n"
            "]",
            m_max_depth,
            m_rr_depth,
            m_train_auto ? "true" : "false",
            (double) m_train_budget,
            (double) m_guide_fraction,
            m_spatial_resolution,
            m_min_tree_records,
            m_rebuild_interval,
            m_max_records_per_cell
        );
    }

    MTS_DECLARE_CLASS()

private:
    void ensure_grid(const Scene *scene) const {
        std::scoped_lock lock(m_grid_mutex);
        if (m_grid_initialized) {
            return;
        }

        m_scene_bbox = scene->bbox();
        if (!m_scene_bbox.valid()) {
            m_scene_bbox = ScalarBoundingBox3f(ScalarPoint3f(-1.f), ScalarPoint3f(1.f));
        }

        for (int axis = 0; axis < 3; ++axis) {
            Scalar extent = m_scene_bbox.max[axis] - m_scene_bbox.min[axis];
            if (!(extent > Scalar(1e-4f))) {
                m_scene_bbox.min[axis] -= Scalar(0.5f);
                m_scene_bbox.max[axis] += Scalar(0.5f);
            }
        }

        size_t cell_count = (size_t) m_spatial_resolution * (size_t) m_spatial_resolution *
                            (size_t) m_spatial_resolution;
        m_cells.clear();
        m_cells.reserve(cell_count);
        for (size_t i = 0; i < cell_count; ++i) {
            m_cells.emplace_back(std::make_unique<SpatialCell>());
        }
        m_grid_initialized = true;
    }

    size_t cell_index(const Point3f &position) const {
        size_t index = 0;
        size_t stride = 1;

        for (int axis = 0; axis < 3; ++axis) {
            Scalar value = position[axis];
            if (!std::isfinite((double) value)) {
                return 0;
            }

            Scalar min_value = m_scene_bbox.min[axis];
            Scalar max_value = m_scene_bbox.max[axis];
            Scalar extent = max(max_value - min_value, Scalar(1e-4f));
            Scalar normalized = clamp((value - min_value) / extent, Scalar(0.f), Scalar(0.999999f));
            size_t coord = std::min<size_t>((size_t) (normalized * (Scalar) m_spatial_resolution),
                                            (size_t) m_spatial_resolution - 1);
            index += coord * stride;
            stride *= (size_t) m_spatial_resolution;
        }

        return index;
    }

    std::shared_ptr<DTreeWrapperType> query_tree(const Point3f &position) const {
        if (!m_grid_initialized || m_cells.empty()) {
            return nullptr;
        }

        const SpatialCell &cell = *m_cells[cell_index(position)];
        std::scoped_lock lock(cell.mutex);
        return cell.tree;
    }

    Scalar mixed_direct_pdf(const SurfaceInteraction3f &si, const Vector3f &world_dir, Float emitter_pdf) const {
        std::shared_ptr<DTreeWrapperType> tree = query_tree(si.p);
        if (tree == nullptr || !(m_guide_fraction > 0.f)) {
            return emitter_pdf;
        }

        Scalar guide_pdf = tree->pdf(world_dir);
        return (Scalar(1.f) - m_guide_fraction) * emitter_pdf + m_guide_fraction * guide_pdf;
    }

    std::shared_ptr<DTreeWrapperType> rebuild_tree_from_records(
        const std::vector<DTreeRecordType> &records) const {
        auto tree = std::make_shared<DTreeWrapperType>();
        for (const DTreeRecordType &record : records) {
            tree->record(record);
        }
        tree->build();
        if (!(tree->meanRadiance() > 0.f)) {
            return nullptr;
        }
        return tree;
    }

    void append_record(SpatialCell &cell, const DTreeRecordType &record) const {
        if (cell.records.size() < m_max_records_per_cell) {
            cell.records.push_back(record);
            return;
        }

        size_t slot = cell.total_records % m_max_records_per_cell;
        cell.records[slot] = record;
    }

    void maybe_rebuild_cell(SpatialCell &cell) const {
        std::vector<DTreeRecordType> snapshot;
        size_t snapshot_total = 0;

        {
            std::scoped_lock lock(cell.mutex);
            size_t pending_records = cell.total_records - cell.last_rebuild_records;
            if (cell.rebuild_in_progress || cell.total_records < m_min_tree_records ||
                pending_records < m_rebuild_interval) {
                return;
            }

            cell.rebuild_in_progress = true;
            snapshot = cell.records;
            snapshot_total = cell.total_records;
        }

        std::shared_ptr<DTreeWrapperType> rebuilt_tree = rebuild_tree_from_records(snapshot);

        {
            std::scoped_lock lock(cell.mutex);
            if (rebuilt_tree != nullptr) {
                cell.tree = rebuilt_tree;
            }
            cell.last_rebuild_records = std::max(cell.last_rebuild_records, snapshot_total);
            cell.rebuild_in_progress = false;
        }
    }

    Scalar spectrum_mean(const Spectrum &value) const {
        Spectrum unpolarized = depolarize(value);
        return (unpolarized[0] + unpolarized[1] + unpolarized[2]) / Scalar(3.f);
    }

    void record_training_direction(const SurfaceInteraction3f &si, const Vector3f &world_dir,
                                   Scalar target_value, Sampler *sampler, Mask active) const {
        if (!m_train_auto || !any_or<true>(active)) {
            return;
        }
        if (!(target_value > 0.f) || !std::isfinite((double) target_value)) {
            return;
        }
        if (sampler->next_1d(active) > m_train_budget) {
            return;
        }

        DTreeRecordType record{};
        record.d = world_dir;
        record.radiance = target_value;
        record.product = target_value;
        record.woPdf = 1.f;
        record.bsdfPdf = 0.f;
        record.dTreePdf = 0.f;
        record.statisticalWeight = 1.f;
        record.isDelta = false;
        record.tau = 0;

        SpatialCell &cell = *m_cells[cell_index(si.p)];
        {
            std::scoped_lock lock(cell.mutex);
            append_record(cell, record);
            cell.total_records += 1;
        }
        maybe_rebuild_cell(cell);
    }

    Spectrum sample_direct_guided(const Scene *scene, Sampler *sampler, const SurfaceInteraction3f &si,
                                  const BSDFPtr &bsdf, const BSDFContext &ctx,
                                  const Spectrum &throughput, Mask active) const {
        std::shared_ptr<DTreeWrapperType> tree = query_tree(si.p);
        Float guide_fraction = (tree != nullptr) ? m_guide_fraction : Float(0.f);
        Float emitter_fraction = 1.f - guide_fraction;

        bool use_guided = tree != nullptr && guide_fraction > 0.f &&
                          sampler->next_1d(active) < guide_fraction;

        if (!use_guided) {
            DirectionSample3f ds;
            Spectrum emitter_val;
            std::tie(ds, emitter_val) = scene->sample_emitter_direction(
                si, sampler->next_2d(active), true, active);
            active &= neq(ds.pdf, 0.f);
            if (none_or<false>(active)) {
                return Spectrum(0.f);
            }

            Vector3f wo = si.to_local(ds.d);
            Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active);
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);
            active &= any(neq(depolarize(bsdf_val), 0.f));
            if (none_or<false>(active)) {
                return Spectrum(0.f);
            }

            Float guide_pdf = (tree != nullptr) ? tree->pdf(ds.d) : Float(0.f);
            Float mixture_pdf = emitter_fraction * ds.pdf + guide_fraction * guide_pdf;
            active &= mixture_pdf > 0.f;
            if (none_or<false>(active)) {
                return Spectrum(0.f);
            }

            Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active);
            Float mis = mis_weight(mixture_pdf, bsdf_pdf);

            Scalar training_value = spectrum_mean(bsdf_val * emitter_val) * ds.pdf;
            record_training_direction(si, ds.d, training_value, sampler, active);

            return mis * throughput * bsdf_val * emitter_val * ds.pdf / mixture_pdf;
        }

        Vector3f world_dir = tree->sample(sampler);
        Vector3f wo = si.to_local(world_dir);
        Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active);
        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);
        active &= any(neq(depolarize(bsdf_val), 0.f));
        if (none_or<false>(active)) {
            return Spectrum(0.f);
        }

        Ray3f direct_ray = si.spawn_ray(world_dir);
        SurfaceInteraction3f si_direct = scene->ray_intersect(direct_ray, active);
        EmitterPtr emitter = si_direct.emitter(scene, active);
        active &= neq(emitter, nullptr);
        if (none_or<false>(active)) {
            return Spectrum(0.f);
        }

        Spectrum emitter_val = emitter->eval(si_direct, active);
        DirectionSample3f ds(si_direct, si);
        ds.object = emitter;

        Float emitter_pdf = scene->pdf_emitter_direction(si, ds, active);
        Float guide_pdf = tree->pdf(world_dir);
        Float mixture_pdf = emitter_fraction * emitter_pdf + guide_fraction * guide_pdf;
        active &= mixture_pdf > 0.f;
        if (none_or<false>(active)) {
            return Spectrum(0.f);
        }

        Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active);
        Float mis = mis_weight(mixture_pdf, bsdf_pdf);

        Scalar training_value = spectrum_mean(bsdf_val * emitter_val);
        record_training_direction(si, world_dir, training_value, sampler, active);

        return mis * throughput * bsdf_val * emitter_val / mixture_pdf;
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

private:
    bool m_train_auto = true;
    Scalar m_train_budget = 1.f;
    Scalar m_guide_fraction = 0.5f;
    int m_samples_per_pass = 1;
    int m_spatial_resolution = 8;
    size_t m_min_tree_records = 128;
    size_t m_rebuild_interval = 64;
    size_t m_max_records_per_cell = 4096;

    mutable std::mutex m_grid_mutex;
    mutable bool m_grid_initialized = false;
    mutable ScalarBoundingBox3f m_scene_bbox;
    mutable std::vector<std::unique_ptr<SpatialCell>> m_cells;
};

MTS_IMPLEMENT_CLASS_VARIANT(PPGIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PPGIntegrator, "Path Guiding baseline integrator");

NAMESPACE_END(mitsuba)
