#include "render_counters.h"
#include <algorithm>

#include "ann.h"
#include "manifold_path_guiding.h"
#include "spatial_structure.h"
#include "dtree.h"
#include "util.h"
#include "chain_distribution.h"
#include "length_proposal_mixture.h"

namespace mitsuba {
int g_iter = 0;
}

#define stat(n, x)                                                                                                     \
    {                                                                                                                  \
        static long long sss = 0;                                                                                      \
        sss += x;                                                                                                      \
        if (rand() % 30000 == 0) {                                                                                     \
            std::cout << n << ": " << sss << std::endl;                                                                \
        }                                                                                                              \
    }

NAMESPACE_BEGIN(mitsuba)


ManifoldPathGuidingConfig global_sms_config;


template <typename Float, typename Spectrum>
static inline ThreadLocal<std::vector<SubpathSample<Float, Spectrum>>> recorded_samples_thread_local;

template <typename Float, typename Spectrum>
static void export_subpath_samples_jsonl(const std::vector<SubpathSample<Float, Spectrum>> &samples,
                                         const std::string &path) {
    if (path.empty()) {
        return;
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        Log(Warn, "Failed to open export_samples_path: %s", path.c_str());
        return;
    }

    out << std::fixed << std::setprecision(9);
    for (const auto &sample : samples) {
        out << "{\"x_d\":[" << (double) sample.xD[0] << "," << (double) sample.xD[1] << ","
            << (double) sample.xD[2] << "],"
            << "\"x_l\":[" << (double) sample.xL[0] << "," << (double) sample.xL[1] << ","
            << (double) sample.xL[2] << "],"
            << "\"x_1\":[" << (double) sample.x1[0] << "," << (double) sample.x1[1] << ","
            << (double) sample.x1[2] << "],"
            << "\"n_d\":[" << (double) sample.nD[0] << "," << (double) sample.nD[1] << ","
            << (double) sample.nD[2] << "],"
            << "\"n_l\":[" << (double) sample.nL[0] << "," << (double) sample.nL[1] << ","
            << (double) sample.nL[2] << "],"
            << "\"d_l\":[" << (double) sample.dL[0] << "," << (double) sample.dL[1] << ","
            << (double) sample.dL[2] << "],"
            << "\"energy\":" << (double) sample.energy << ","
            << "\"bounce\":" << sample.bounce << ","
            << "\"chain_type\":" << sample.type << ","
            << "\"filtered\":" << (sample.filtered ? "true" : "false") << "}"
            << std::endl;
    }
    std::cout << "Exported subpath samples " << samples.size() << " -> " << path << std::endl;
}

// ===============================
// Manifold Sampler
// ===============================
template <typename Float, typename Spectrum> class GuidedManifoldSampler {
public:
    MTS_IMPORT_TYPES(BSDF, Sampler, Scene, Shape);
    using BSDFPtr            = typename RenderAliases::BSDFPtr;
    using ShapePtr           = typename RenderAliases::ShapePtr;
    using ManifoldVertex     = ::mitsuba::ManifoldVertex<Float, Spectrum>;
    using EmitterInteraction = ::mitsuba::EmitterInteraction<Float, Spectrum>;
    using SpecularManifold   = ::mitsuba::SpecularManifold<Float, Spectrum>;
    using Manifold_Walk      = ::mitsuba::Manifold_Walk<Float, Spectrum>;
    using SubpathSample      = ::mitsuba::SubpathSample<Float, Spectrum>;

    using SpatialStructure                 = ::mitsuba::SpatialStructure<Float, Spectrum>;
    using SpatialStructureANN              = ::mitsuba::SpatialStructureANN<Float, Spectrum>;
    using SpatialStructureSTree            = ::mitsuba::SpatialStructureSTree<Float, Spectrum>;
    using ChainDistribution                = ::mitsuba::ChainDistribution<Float, Spectrum>;
    using ChainDistributionSamplingContext = ::mitsuba::ChainDistributionSamplingContext<Float, Spectrum>;
    using MaskedDiffusionRuntimeModel      = ::mitsuba::MaskedDiffusionRuntimeModel<Float>;
    using MaskedDiffusionPrediction        = typename MaskedDiffusionRuntimeModel::Prediction;
    using MaskedDiffusionScalar            = typename MaskedDiffusionRuntimeModel::Scalar;
    using MaskedDiffusionOrderedDescriptor =
        typename MaskedDiffusionRuntimeModel::OrderedDescriptorSample;

    using MySTreeNode = ::mitsuba::QuadTreeNode<Float, Spectrum>;

    int sample_bounce     = -1;
    float inv_prob_bounce = 1.0f;
    ref<Shape> sampled_shape;

    GuidedManifoldSampler() {}

    GuidedManifoldSampler(const Scene *scene, const ManifoldPathGuidingConfig &config, const SurfaceInteraction3f si,
                          const std::vector<SubpathSample> *subpath_sample_list,
                          const std::vector<SubpathSample> *subpath_sample_list_ext,
                          const std::shared_ptr<SpatialStructure> spatial_structure,
                          const std::shared_ptr<SpatialStructure> spatial_structure_ext,
                          std::shared_ptr<MaskedDiffusionRuntimeModel> masked_diffusion_runtime) {
        init(scene, config, si, EmitterInteraction(), subpath_sample_list, subpath_sample_list_ext, spatial_structure,
             spatial_structure_ext,
             masked_diffusion_runtime);
    }

    void init(const Scene *_scene, const ManifoldPathGuidingConfig &_config, const SurfaceInteraction3f _si,
              const EmitterInteraction _ei, const std::vector<SubpathSample> *_subpath_sample_list,
              const std::vector<SubpathSample> *_subpath_sample_list_ext,
              std::shared_ptr<SpatialStructure> _spatial_structure,
              std::shared_ptr<SpatialStructure> _spatial_structure_ext,
              std::shared_ptr<MaskedDiffusionRuntimeModel> _masked_diffusion_runtime) {
        si = _si, ei = _ei, m_scene = _scene, m_config = _config, manifold_walk = Manifold_Walk(_scene, _config),
        subpath_sample_list = _subpath_sample_list, subpath_sample_list_ext = _subpath_sample_list_ext,
        spatial_structure = _spatial_structure, spatial_structure_ext = _spatial_structure_ext,
        masked_diffusion_runtime = _masked_diffusion_runtime;
        // This sampler object is reused between endpoint queries. Never let a
        // distribution from an earlier query leak into a later unguided one.
        chain_distr = nullptr;
        chain_distr_ext = nullptr;
        if (m_config.guided || (m_config.train_auto && m_online_iteration)) {
            NanosecondTimer timer;
            chain_distr = spatial_structure ? spatial_structure->query(si.p, ei.p) : nullptr;
            distr_ctx.clear();
            distr_ctx.si = si;
            // Queries may land on empty leaves; verify usability before initializing the product distribution.
            if (m_config.product_sampling && has_usable_guidance_distribution())
                chain_distr->init_product(distr_ctx);
            perf_time_guide += timer.value();
        }
        if (m_config.initial == 2) {
            chain_distr_ext = spatial_structure_ext ? spatial_structure_ext->query(si.p, 0) : nullptr;
        }

        masked_diffusion_prediction = MaskedDiffusionPrediction();
        ordered_descriptor = MaskedDiffusionOrderedDescriptor();
        ordered_descriptor_cached = false;
        if (masked_diffusion_runtime && masked_diffusion_runtime->valid()) {
            masked_diffusion_prediction = masked_diffusion_runtime->predict(
                build_condition_features(),
                m_config.masked_diffusion_steps
            );
        }
    }

    ~GuidedManifoldSampler() {}

    bool has_usable_guidance_distribution() const {
        // Fall back to unguided sampling when training samples and leaf-local support are both too low,
        // avoiding unstable sparse distributions early on.
        constexpr size_t kMinGlobalGuidanceSamples = 256;
        constexpr Float kMinGuidanceWeight = (Float) 8.f;
        if (!subpath_sample_list || subpath_sample_list->size() < kMinGlobalGuidanceSamples) {
            return false;
        }
        return chain_distr && chain_distr->valid() && chain_distr->statisticalWeight() >= kMinGuidanceWeight;
    }

    std::vector<scalar_t<Float>> build_condition_features() const {
        using Scalar = scalar_t<Float>;
        // Feature order must exactly match flatten_condition in the Python training pipeline.
        Vector3f delta = ei.p - si.p;
        Scalar distance = norm(delta);
        std::vector<Scalar> condition;
        Vector3f delta_dir = distance > (Scalar) 1e-6f ? delta / distance : Vector3f(0.f);
        Scalar receiver_alignment = dot(delta_dir, si.n);
        Scalar emitter_alignment = dot(-delta_dir, ei.n);
        Scalar emission_alignment = dot(delta_dir, ei.d);

        condition.reserve(22);
        condition.push_back((Scalar) si.p[0]);
        condition.push_back((Scalar) si.p[1]);
        condition.push_back((Scalar) si.p[2]);
        condition.push_back((Scalar) ei.p[0]);
        condition.push_back((Scalar) ei.p[1]);
        condition.push_back((Scalar) ei.p[2]);
        condition.push_back((Scalar) delta[0]);
        condition.push_back((Scalar) delta[1]);
        condition.push_back((Scalar) delta[2]);
        condition.push_back(distance);
        condition.push_back((Scalar) si.n[0]);
        condition.push_back((Scalar) si.n[1]);
        condition.push_back((Scalar) si.n[2]);
        condition.push_back((Scalar) ei.n[0]);
        condition.push_back((Scalar) ei.n[1]);
        condition.push_back((Scalar) ei.n[2]);
        condition.push_back((Scalar) ei.d[0]);
        condition.push_back((Scalar) ei.d[1]);
        condition.push_back((Scalar) ei.d[2]);
        // Explicit alignment scalars encode geometry such as whether surfaces face the path and the emitter faces the receiver.
        condition.push_back(receiver_alignment);
        condition.push_back(emitter_alignment);
        condition.push_back(emission_alignment);
        return condition;
    }

    bool has_masked_diffusion_prediction() const {
        return masked_diffusion_runtime &&
               (masked_diffusion_prediction.valid ||
                (masked_diffusion_runtime->ordered_descriptor_mode() &&
                 masked_diffusion_prediction.ordered_context_valid));
    }

    bool can_cache_ordered_descriptor() const {
        if (!masked_diffusion_runtime ||
            !masked_diffusion_runtime->ordered_descriptor_mode() ||
            !masked_diffusion_prediction.ordered_context_valid) {
            return false;
        }

        // The existing MPG/photon fallbacks form an implicit proposal mixture
        // whose shared basin mass is estimated by re-hit. Factoring only the
        // ordered component out of that mixture would be incorrect, so use the
        // exact discrete factorization only for the pure neural proposal.
        if (has_usable_guidance_distribution() || m_config.initial == 2) {
            return false;
        }

        // Ablation 2 deliberately ignores the sampled type descriptor, so its
        // latent value is not a disjoint proposal stratum and cannot receive an
        // independent inverse-probability factor. Forced tau likewise retains
        // its historical semantics.
        return m_config.ablation != 2 && m_config.sms_tau_force < 0;
    }

    void prepare_ordered_descriptor(ref<Sampler> sampler) {
        ordered_descriptor = MaskedDiffusionOrderedDescriptor();
        ordered_descriptor_cached = false;
        if (!can_cache_ordered_descriptor() || sample_bounce <= 0) {
            return;
        }

        std::vector<MaskedDiffusionScalar> bit_uniforms;
        bit_uniforms.reserve((size_t) sample_bounce);
        for (int bit = 0; bit < sample_bounce; ++bit) {
            bit_uniforms.push_back((MaskedDiffusionScalar) sampler->next_1d());
        }
        MaskedDiffusionOrderedDescriptor sampled =
            masked_diffusion_runtime->sample_ordered_descriptor_for_length(
                masked_diffusion_prediction,
                sample_bounce,
                bit_uniforms
            );
        if (!sampled.valid ||
            !std::isfinite((double) sampled.conditional_log_probability) ||
            sampled.conditional_log_probability > (MaskedDiffusionScalar) 0.f) {
            return;
        }

        ordered_descriptor = std::move(sampled);
        ordered_descriptor_cached = true;
        perf_diag_ordered_descriptor_cached++;
    }

    std::pair<Vector3f, int> sample_seed_direction_and_tau(ref<Sampler> sampler) {
        Point3f x0         = si.p;
        bool succeed       = false;
        Vector3f guide_dir = 0;
        // When vanilla SMS needs a fixed chain type, inject tau directly from configuration.
        int tau            = m_config.sms_tau_force;

        if (has_usable_guidance_distribution() && chain_distr->valid_bounce(sample_bounce) &&
            ((m_config.guided && sampler->next_1d() > m_config.prob_uniform) ||
             (m_online_iteration && !m_online_last_iteration && sampler->next_1d() > m_config.prob_uniform_train) ||
             (m_online_iteration && m_online_last_iteration && sampler->next_1d() > m_config.prob_uniform))) {
            NanosecondTimer timer;
            tau       = chain_distr->sample_tau(sample_bounce, sampler, distr_ctx);
            guide_dir = chain_distr->sample_omega(tau, sampler, distr_ctx);

            if (dot(guide_dir, si.n) > 0) {
                succeed = true;
            }
            perf_time_guide += timer.value();
        }

        // Use distribution from the external samples (e.g., photons)
        if (!succeed && m_config.initial == 2 && sampler->next_1d() > 0.5f) {
            guide_dir = chain_distr_ext->sample_omega(-1, sampler, distr_ctx);
            if (dot(guide_dir, si.n) > 0) {
                succeed = true;
            }
        }

        if (!succeed && has_masked_diffusion_prediction()) {
            if (masked_diffusion_runtime->ordered_descriptor_mode()) {
                MaskedDiffusionOrderedDescriptor transient_sample;
                const MaskedDiffusionOrderedDescriptor *ordered_sample = nullptr;
                if (ordered_descriptor_cached &&
                    ordered_descriptor.chain_length == sample_bounce) {
                    ordered_sample = &ordered_descriptor;
                    perf_diag_ordered_descriptor_uses++;
                } else {
                    // Hybrid/ablation paths retain the original full-procedure
                    // re-hit estimator because their proposal components cannot
                    // be factored independently without a mixture PDF.
                    std::vector<MaskedDiffusionScalar> bit_uniforms;
                    bit_uniforms.reserve((size_t) std::max(sample_bounce, 0));
                    for (int bit = 0; bit < sample_bounce; ++bit) {
                        bit_uniforms.push_back((MaskedDiffusionScalar) sampler->next_1d());
                    }
                    transient_sample =
                        masked_diffusion_runtime->sample_ordered_descriptor_for_length(
                            masked_diffusion_prediction,
                            sample_bounce,
                            bit_uniforms
                        );
                    ordered_sample = &transient_sample;
                }
                if (ordered_sample->valid) {
                    if (m_config.ablation != 2 && m_config.sms_tau_force < 0) {
                        tau = ordered_sample->tau;
                    }
                    if (m_config.ablation != 3) {
                        guide_dir = masked_diffusion_runtime->sample_direction(
                            ordered_sample->direction_prediction,
                            sampler->next_1d(),
                            sampler->next_1d(),
                            sampler->next_1d(),
                            (MaskedDiffusionScalar) m_config.learned_direction_kappa_scale
                        );
                        if (should_flip_learned_direction(
                                m_config.learned_direction_flip_to_receiver_hemisphere,
                                dot(guide_dir, si.n))) {
                            guide_dir *= -1.f;
                        }
                        if (dot(guide_dir, guide_dir) > 1e-12f) {
                            succeed = true;
                        }
                    }
                }
            } else {
                // Runtime fallback reuses the original ablation semantics so the main method and ablations stay consistent.
                // NPM-adapted predicts direction only. Its chain length uses the
                // shared geometric policy and tau remains -1 so the existing
                // recurrent trace infers the realized chain type. Never invent
                // descriptor logits for a direction-only checkpoint.
                if (m_config.ablation != 2 &&
                    !masked_diffusion_runtime->direction_only_mode()) {
                    tau = masked_diffusion_runtime->predict_tau(masked_diffusion_prediction, sample_bounce);
                }
                if (m_config.ablation != 3) {
                    guide_dir = masked_diffusion_runtime->sample_direction(
                        masked_diffusion_prediction,
                        sampler->next_1d(),
                        sampler->next_1d(),
                        sampler->next_1d(),
                        (MaskedDiffusionScalar) m_config.learned_direction_kappa_scale
                    );
                    // Preserve the legacy receiver-hemisphere fold only when requested.
                    if (should_flip_learned_direction(
                            m_config.learned_direction_flip_to_receiver_hemisphere,
                            dot(guide_dir, si.n))) {
                        guide_dir *= -1.f;
                    }
                    if (dot(guide_dir, guide_dir) > 1e-12f) {
                        succeed = true;
                    }
                }
            }
        }

        // Uniform
        if (!succeed) {
            if (m_config.initial != 1) {
                std::vector<ref<Shape>> shapes;
                const auto &multi_casters  = m_scene->caustic_casters_multi_scatter();
                const auto &single_casters = m_scene->caustic_casters_single_scatter();
                shapes.insert(shapes.end(), multi_casters.begin(), multi_casters.end());
                shapes.insert(shapes.end(), single_casters.begin(), single_casters.end());

                if (shapes.empty()) {
                    shapes.reserve(m_scene->shapes().size());
                    for (auto &s : m_scene->shapes()) {
                        BSDFPtr b = s->bsdf();
                        if (!b)
                            continue;
                        auto f = b->flags();
                        if (has_flag(f, BSDFFlags::Delta) ||
                            has_flag(f, BSDFFlags::DeltaReflection) ||
                            has_flag(f, BSDFFlags::DeltaTransmission)) {
                            shapes.push_back(s);
                        }
                    }
                }

                // If no specular (delta) shapes exist, fall back to uniform sphere sampling (same as the initial==1 branch).
                if (shapes.empty()) {
                    guide_dir                = warp::square_to_uniform_sphere(sampler->next_2d());
                    SurfaceInteraction3f its = m_scene->ray_intersect(si.spawn_ray(guide_dir));
                    sampled_shape            = (Shape *) its.shape;
                    if (dot(guide_dir, si.n) < 0) {
                        guide_dir = -guide_dir;
                    }
                } else {
                    ref<Shape> shape_ = shapes[(size_t) (sampler->next_1d() * shapes.size())];
                    sampled_shape     = shape_;
                    PositionSample3f ps = shape_->sample_position(si.time, sampler->next_2d());
                    Point3f x1          = ps.p;
                    guide_dir           = normalize(x1 - x0);
                }

            } else {
                guide_dir                = warp::square_to_uniform_sphere(sampler->next_2d());
                SurfaceInteraction3f its = m_scene->ray_intersect(si.spawn_ray(guide_dir));
                sampled_shape            = (Shape *) its.shape;
                if (dot(guide_dir, si.n) < 0) {
                    guide_dir *= -1;
                }
            }
            // We wait the tau sample till the recurrent ray-tracing phase and return tau = -1.
        }
        return { guide_dir, tau };
    }

    Mask sample_path(const SurfaceInteraction3f &si, const EmitterInteraction &ei, ref<Sampler> sampler) {
        Mask success_seed = sample_seed_path(sampler);
        if (!success_seed || m_current_path.empty()) {
            return false;
        }
        Mask success_solve      = manifold_walk.newton_solver(si, ei, m_current_path, m_offset_normals);
        if (!success_solve || m_current_path.empty() || (int) m_current_path.size() < sample_bounce) {
            return false;
        }

        return true;
    }

    Mask sample_seed_path(ref<Sampler> sampler) {
        m_current_path.clear();
        SurfaceInteraction3f si_(si);
        Point3f x0     = si_.p;
        auto [wo, tau] = sample_seed_direction_and_tau(sampler);

        Ray3f ray(x0, wo, si_.time, si_.wavelengths);
        while (true) {
            int bounce = m_current_path.size();
            if (bounce >= m_seed_path.size()) {
                break;
            }
            si_ = m_scene->ray_intersect(ray);
            if (!si_.is_valid()) {
                return false;
            }
            const ShapePtr shape = si_.shape;
            BSDFPtr bsdf = shape->bsdf();
            if (!bsdf) {
                return false;
            }
            auto f = bsdf->flags();
            if (!(has_flag(f, BSDFFlags::Delta) ||
                  has_flag(f, BSDFFlags::DeltaReflection) ||
                  has_flag(f, BSDFFlags::DeltaTransmission))) {
                // We intersected something that cannot be part of a specular chain
                return false;
            }


            if (shape != m_seed_path[bounce].shape) {
                /* When we sample multiple paths for the PDF estimation process,
                be conservative and only allow paths that intersect the same
                shapes again. */
                return false;
            }

            // On first intersect, make sure that we actually hit this target shape (for surface sampling)
            if (!((m_config.guided || m_config.train_auto) || m_online_iteration) && bounce == 0 &&
                (shape != sampled_shape && m_config.sms_strict_uniform)) {
                return false;
            }

            // Create the path vertex
            ManifoldVertex vertex = ManifoldVertex(si_, 0.f);
            m_current_path.push_back(vertex);
            Vector3f n_offset = m_offset_normals[bounce];

            Vector3f m           = vertex.s * n_offset[0] + vertex.t * n_offset[1] + vertex.n * n_offset[2];
            Vector3f wi          = -wo;
            bool scatter_success = false;

            float reflect_prob = 1.f;
            if (vertex.eta != 1.f) { // dielectric
                if (tau != -1) {
                    // guided type sampling
                    if (get_chaintype_bit(tau, bounce) == 1) {
                        reflect_prob = 0.0;
                    } else {
                        reflect_prob = 1.0;
                    }
                } else {
                    // Fresnel type sampling
                    auto avg              = [&](Spectrum s) { return (s[0] + s[1] + s[2]) / 3; };
                    const BSDFPtr bsdf    = si_.bsdf();
                    Complex<Spectrum> ior = si_.bsdf()->ior(si_);
                    Spectrum f;
                    Frame3f frame   = bsdf->frame(si, 0.f);
                    Float cos_theta = dot(frame.n, wi);
                    Float eta       = hmean(real(ior));
                    if (cos_theta < 0.f) {
                        eta = rcp(eta);
                    }
                    auto [F, unused_0, unused_1, unused_2] = fresnel(cos_theta, eta);
                    reflect_prob                           = avg(F);
                }
                if (m_config.sms_type_force == 1) {
                    reflect_prob = 0.0;
                }
            }
            // ! We disregard both tau and sms_type_force for conductor.

            if (sampler->next_1d() < reflect_prob) {
                std::tie(scatter_success, wo)    = SpecularManifold::reflect(wi, m);
                m_current_path.back().is_refract = false;
            } else {
                std::tie(scatter_success, wo)    = SpecularManifold::refract(wi, m, vertex.eta);
                m_current_path.back().is_refract = true;
            }
            if (!scatter_success) {
                // We must have hit total internal reflection. Abort.
                return false;
            }
            ray = si_.spawn_ray(wo);
        }

        return true;
    }

    int getTau(std::vector<ManifoldVertex> &v) {
        if (sample_bounce <= 0 || (int) v.size() < sample_bounce) {
            return -1;
        }
        int tau = 0;
        set_bit(tau, sample_bounce);
        for (int i = 0; i < sample_bounce; i++) {
            set_chaintype_bit(tau, i, v[i].is_refract);
        }
        return tau;
    }

    Float clamp_learned_proposal_inv_prob(Float inv_prob) const {
        const Float limit = (Float) m_config.learned_proposal_inv_prob_clamp;
        if (!(limit > 0.f)) {
            return inv_prob;
        }
        if (!std::isfinite((double) inv_prob) || inv_prob < 0.f) {
            return 0.f;
        }
        if (inv_prob > limit) rb::lc.inverse_cap++;
        return std::min(inv_prob, limit);
    }

    Float finalize_seed_inverse_probability(Float conditional_inverse) const {
        rb::lc.inverse_weights++;
        // A zero value is the existing max-trial failure sentinel. Do not let
        // the log-domain calculation below resurrect a rejected estimate.
        if (!std::isfinite((double) conditional_inverse) ||
            conditional_inverse <= (Float) 0.f) {
            return (Float) 0.f;
        }
        if (!ordered_descriptor_cached) {
            return clamp_learned_proposal_inv_prob(conditional_inverse);
        }

        // Length is corrected separately by inv_prob_bounce.  Here the exact
        // ordered factor is only q(type bits | selected length).  Work in log
        // space so a rare sampled descriptor cannot overflow before an
        // explicitly configured practical clamp is applied.
        const double log_joint_inverse =
            std::log(std::max((double) conditional_inverse, 1e-300)) -
            (double) ordered_descriptor.conditional_log_probability;
        const double configured_limit =
            (double) m_config.learned_proposal_inv_prob_clamp;
        if (configured_limit > 0.0 &&
            log_joint_inverse >= std::log(configured_limit)) {
            rb::lc.inverse_cap++; return (Float) configured_limit;
        }

        const double float_limit = (double) std::numeric_limits<scalar_t<Float>>::max();
        if (log_joint_inverse >= std::log(float_limit)) {
            return std::numeric_limits<scalar_t<Float>>::infinity();
        }
        return clamp_learned_proposal_inv_prob(
            (Float) std::exp(log_joint_inverse)
        );
    }

    Spectrum clamp_learned_proposal_contribution(Spectrum value) const {
        rb::lc.contributions++;
        const Float limit = (Float) m_config.learned_proposal_contribution_clamp;
        if (!(limit > 0.f)) {
            return value;
        }
        Float peak = hmax(depolarize(value));
        if (!std::isfinite((double) peak) || peak < 0.f) {
            return 0.f;
        }
        if (peak > limit) {
            rb::lc.contribution_cap++;
            value *= limit / peak;
        }
        return value;
    }

    void maybe_record_seed_candidate(
        const std::vector<ManifoldVertex> &sampled_path,
        const EmitterInteraction &ei,
        ref<Sampler> sampler
    ) {
        if (!m_config.export_seed_candidates || sampled_path.empty() || sample_bounce <= 0) {
            return;
        }
        const Float probability = (Float) m_config.export_seed_candidate_probability;
        if (probability <= 0.f || sampler->next_1d() > probability) {
            return;
        }
        const int max_records = m_config.export_seed_candidate_max;
        if (max_records > 0 && perf_diag_candidate_stored.load() >= (unsigned long long) max_records) {
            perf_diag_candidate_limit++;
            return;
        }

        std::vector<ManifoldVertex> path_copy = sampled_path;
        int tau = getTau(path_copy);
        if (tau < 0) {
            return;
        }

        SubpathSample new_cache_info;
        new_cache_info.xD       = si.p;
        new_cache_info.xL       = ei.p;
        new_cache_info.x1       = sampled_path[0].p;
        new_cache_info.nD       = si.n;
        new_cache_info.nL       = ei.n;
        new_cache_info.dL       = ei.d;
        new_cache_info.energy   = 1.f;
        new_cache_info.bounce   = sample_bounce;
        new_cache_info.type     = tau;
        new_cache_info.filtered = true;
        std::vector<SubpathSample> &new_data =
            (std::vector<SubpathSample> &) recorded_samples_thread_local<Float, Spectrum>;
        new_data.push_back(new_cache_info);
        perf_diag_candidate_stored++;
    }

    Spectrum specular_manifold_sampling_impl(ref<Sampler> sampler) {
        // Learned bounce selection applies only starting from the first specular vertex.
        if (sample_bounce <= 0) {
            perf_diag_invalid_bounce++;
            return 0.f;
        }

        Point3f x0     = si.p;
        auto [wo, tau] = sample_seed_direction_and_tau(sampler);

        Vector3f initial_direction = wo;
        Ray3f ray(x0, wo, si.time, si.wavelengths);
        SurfaceInteraction3f next_interaction = m_scene->ray_intersect(ray);
        // Runtime directions may miss; validate the intersection before reading the BSDF.
        if (!next_interaction.is_valid()) {
            perf_diag_next_invalid++;
            return 0.f;
        }
        auto bsdf_next = next_interaction.bsdf();
        bool next_is_delta = bsdf_next && has_flag(bsdf_next->flags(), BSDFFlags::Delta);
        if (!next_is_delta) {
            // Early exit: the next interaction is not specular/delta -> no specular manifold chain to start from.
            perf_diag_next_non_delta++;
            return 0.f;
         }
        perf_diag_seed_started++;


        std::vector<ManifoldVertex> sampled_path;
        m_offset_normals.clear();
        m_offset_normals_pdf  = 1.0;
        Spectrum specular_val = 0.f;
        SurfaceInteraction3f si_(si);
        while (true) {
            int bounce = sampled_path.size();
            if (bounce > sample_bounce) {
                break;
            }
            if (bounce == sample_bounce) {
                m_current_path          = sampled_path;
                Mask success            = manifold_walk.newton_solver(si, ei, m_current_path,
                                                                      m_offset_normals); // Newton Solver will change current_path
                if (!success || m_current_path.empty() || (int) m_current_path.size() < sample_bounce) {
                    perf_diag_newton_fail++;
                    maybe_record_seed_candidate(sampled_path, ei, sampler);
                    break;
                }

                ManifoldVertex vtx_last = m_current_path[m_current_path.size() - 1];
                auto [success_e, vy] =
                    SpecularManifold::emitter_interaction_to_vertex(m_scene, ei, vtx_last.p, si.time, si.wavelengths);
                if (success && success_e) {
                    perf_diag_emitter_success++;
                    m_seed_path                    = m_current_path;
                    std::vector<ManifoldVertex> sampled_solution_path = m_current_path;
                    Point3f x1                     = sampled_solution_path[0].p;
                    Vector3f direction             = normalize(sampled_solution_path[0].p - si.p);
                    std::vector<ManifoldVertex> &v = sampled_solution_path;
                    int sampled_tau                = getTau(sampled_solution_path);
                    if (sampled_tau < 0) {
                        perf_diag_bad_tau++;
                        break;
                    }
                    // Make descriptor strata disjoint before applying the
                    // explicit 1/q(bits | length) factor. This matters for
                    // events such as conductor interactions where a requested
                    // type bit can otherwise be ignored by path construction.
                    if (ordered_descriptor_cached &&
                        sampled_tau != ordered_descriptor.tau) {
                        perf_diag_ordered_descriptor_mismatch++;
                        break;
                    }

                    // * EVAL
                    NanosecondTimer timer_eval;
                    Spectrum path_contribution =
                        SpecularManifold::evaluate_path_contribution(m_scene, sampled_solution_path, si, ei);
                    perf_time_eval += timer_eval.value();
                    BSDFContext ctx;
                    Spectrum bsdf_val       = si.bsdf()->eval(ctx, si, si.to_local(direction));
                    int iterations          = 1;
                    unsigned long long rb_retries=0;
                    Float inv_prob_estimate = 0.0;
                    int repeat_bernoulli    = global_sms_config.rep_ber;
                    if (repeat_bernoulli < 0) {
                        // means repeating during the rendering phase only
                        repeat_bernoulli *= -1;
                        if (m_online_last_iteration == false) {
                            repeat_bernoulli = 1;
                        }
                    }
                    NanosecondTimer timer;
                    for (int _ = 0; _ < repeat_bernoulli; _++) {
                        inv_prob_estimate += 1;
                        while (true) {
                            ScopedPhase scope_phase(ProfilerPhase::SMSCausticsBernoulliTrials);
                            ++rb_retries;
                            bool success_trial = sample_path(si, ei, sampler);

                            if (success_trial) {
                                if ((int) m_current_path.size() < sample_bounce) {
                                    continue;
                                }
                                Vector3f direction_trial = normalize(m_current_path[0].p - si.p);
                                int trial_tau            = getTau(m_current_path);
                                if (trial_tau < 0) {
                                    continue;
                                }

                                if (trial_tau == sampled_tau &&
                                    abs(dot(direction, direction_trial) - 1.f) < m_config.uniqueness_threshold) {
                                    break;
                                }
                            }
                            inv_prob_estimate += 1.f;
                            iterations++;

                            if (m_config.max_trials > 0 && iterations > m_config.max_trials * repeat_bernoulli) {
                                /* There is a tiny chance always to sample super
                                weird paths that will never occur again due to small
                                numerical imprecisions. So setting a (super
                                conservative) threshold here can help to avoid
                                infinite loops. */
                                inv_prob_estimate = 0.f;
                                perf_diag_bernoulli_fail++;
                                break;
                            }
                        }
                    }
                    rb::record_retries(rb_retries);
                    inv_prob_estimate /= repeat_bernoulli;
                    inv_prob_estimate = finalize_seed_inverse_probability(
                        inv_prob_estimate
                    );
                    perf_time_bernoulli += timer.value();
                    if (!std::isfinite((double) m_offset_normals_pdf) || m_offset_normals_pdf <= (Float) 1e-12f) {
                        perf_diag_bad_offset_pdf++;
                        break;
                    }
                    Spectrum manifold_contribution =
                        bsdf_val * path_contribution * inv_prob_estimate / m_offset_normals_pdf;
                    specular_val += clamp_learned_proposal_contribution(manifold_contribution);

                    // * STORE

                    Spectrum result = path_contribution * inv_prob_estimate / m_offset_normals_pdf;
                    auto avg        = [&](Spectrum s) { return (s[0] + s[1] + s[2]) / 3; };

                    Float energy = avg(result);

                    int tau = 0;
                    set_bit(tau, sample_bounce);
                    for (int i = 0; i < sample_bounce; i++) {
                        set_chaintype_bit(tau, i, v[i].is_refract);
                    }
                    float eps = 1e-6;

                    if (energy > eps) {
                        SubpathSample new_cache_info;
                        new_cache_info.xD     = si.p;
                        new_cache_info.xL     = ei.p;
                        new_cache_info.x1     = x1;
                        new_cache_info.nD     = si.n;
                        new_cache_info.nL     = ei.n;
                        new_cache_info.dL     = ei.d;
                        new_cache_info.energy = energy * inv_prob_bounce;
                        new_cache_info.bounce = bounce;
                        new_cache_info.type   = tau;
                        std::vector<SubpathSample> &new_data =
                            (std::vector<SubpathSample> &) recorded_samples_thread_local<Float, Spectrum>;
                        new_data.push_back(new_cache_info);
                        perf_diag_stored++;
                    } else {
                        perf_diag_energy_low++;
                    }
                } else {
                    perf_diag_emitter_fail++;
                }
            }

            si_ = m_scene->ray_intersect(ray);
            if (!si_.is_valid()) {
                perf_diag_chain_invalid++;
                break;
            }
            const ShapePtr shape = si_.shape;
            BSDFPtr bsdf = shape->bsdf();
            if (!bsdf) {
                perf_diag_chain_no_bsdf++;
                break;
            }
            auto f = bsdf->flags();
            if (!(has_flag(f, BSDFFlags::Delta) ||
                  has_flag(f, BSDFFlags::DeltaReflection) ||
                  has_flag(f, BSDFFlags::DeltaTransmission))) {
                // We intersected something that cannot be part of a specular chain
                perf_diag_chain_non_delta++;
                break;
            }


            // On first intersect, make sure that we actually hit this target shape (for surface sampling).
            if (!(m_config.guided || m_online_iteration) && bounce == 0 &&
                (shape != sampled_shape && m_config.sms_strict_uniform)) {
                return false;
            }

            // Create the path vertex
            ManifoldVertex vertex = ManifoldVertex(si_, 0.f);
            sampled_path.push_back(vertex);

            // Potentially sample an offset normal here.
            Vector3f n_offset;
            n_offset    = Vector3f(0.f, 0.f, 1.f);
            Float p_o   = 1.f;
            Float alpha = shape->bsdf()->roughness();
            if (alpha > 0.f) {
                // Assume isotropic Beckmann distribution for the surface
                // roughness.
                MicrofacetDistribution<Float, Spectrum> distr(MicrofacetType::Beckmann, alpha, alpha, false);
                std::tie(n_offset, p_o) = distr.sample(Vector3f(0.f, 0.f, 1.f), sampler->next_2d());
            }
            m_offset_normals_pdf *= p_o;
            m_offset_normals.push_back(n_offset);

            // Perform scattering at vertex, unless we are doing the
            // straight-line MNEE initialization
            Vector3f m = vertex.s * n_offset[0] + vertex.t * n_offset[1] + vertex.n * n_offset[2];

            Vector3f wi          = -wo;
            bool scatter_success = false;

            float reflect_prob = 1.f;
            if (vertex.eta != 1.f) { // dielectric
                if (tau != -1) {
                    if (get_chaintype_bit(tau, bounce) == 1) {
                        reflect_prob = 0.0;
                    } else {
                        reflect_prob = 1.0;
                    }
                } else {
                    auto avg              = [&](Spectrum s) { return (s[0] + s[1] + s[2]) / 3; };
                    const BSDFPtr bsdf    = si_.bsdf();
                    Complex<Spectrum> ior = si_.bsdf()->ior(si_);
                    Spectrum f;
                    Frame3f frame   = bsdf->frame(si, 0.f);
                    Float cos_theta = dot(frame.n, wi);
                    Float eta       = hmean(real(ior));
                    if (cos_theta < 0.f) {
                        eta = rcp(eta);
                    }
                    auto [F, unused_0, unused_1, unused_2] = fresnel(cos_theta, eta);
                    reflect_prob                           = avg(F);
                }
                if (m_config.sms_type_force == 1) {
                    reflect_prob = 0.0;
                }
            }

            if (sampler->next_1d() < reflect_prob) {
                std::tie(scatter_success, wo)  = SpecularManifold::reflect(wi, m);
                sampled_path.back().is_refract = false;
            } else {
                std::tie(scatter_success, wo)  = SpecularManifold::refract(wi, m, vertex.eta);
                sampled_path.back().is_refract = true;
            }

            if (!scatter_success) {
                // We must have hit total internal reflection. Abort.
                perf_diag_scatter_fail++;
                break;
            }
            ray = si_.spawn_ray(wo);
        }
        return specular_val;
    }

    Spectrum specular_manifold_sampling(ref<Sampler> sampler) {
        if (unlikely(!si.is_valid())) {
            return 0.f;
        }
        NanosecondTimer timer;

        std::vector<ShapePtr> shapes;
        shapes.reserve(m_scene->shapes().size());

        for (auto &s : m_scene->shapes()) {
            BSDFPtr b = s->bsdf();
            if (!b)
                continue;
            auto f = b->flags();
            if (has_flag(f, BSDFFlags::Delta) ||
                has_flag(f, BSDFFlags::DeltaReflection) ||
                has_flag(f, BSDFFlags::DeltaTransmission)) {
                shapes.push_back(s);
            }
        }

        if (unlikely(shapes.empty())) {
            return 0.f;
        }


        DiscreteDistribution<Float> geometric_distr;
        std::vector<Float> a;
        a.resize(MAX_CHAIN_LENGTH);
        float acc = 1.0f;
        for (int i = 0; i < MAX_CHAIN_LENGTH; i++) {
            a[i] = acc;
            if (i + 1 > m_sms_rr_depth) {
                acc *= 0.95; // hard coded q
            }
            if (i + 1 > m_sms_max_depth) {
                break;
            }
        }
        geometric_distr = DiscreteDistribution<Float>(a.data(), a.size());

        auto sample_geometric_distr = [&](float r) -> int { return geometric_distr.sample(r); };
        auto pdf_geometric_distr    = [&](int x) -> float { return geometric_distr.eval_pmf_normalized(x); };

        float bounce_alpha = m_config.bounce_alpha;

        if (m_config.ablation == 1) {
            bounce_alpha = 1.0;
        }

        // Guidance queries may yield null or an empty distribution; fall back to the geometric distribution and never dereference null.
        const bool has_guidance_distribution = has_usable_guidance_distribution();

        if (m_config.sms_tau_force >= 0) {
            sample_bounce   = get_bit(m_config.sms_tau_force);
            inv_prob_bounce = 1.0;
        } else if (m_config.sms_bounce_force > 0) {
            sample_bounce   = m_config.sms_bounce_force;
            inv_prob_bounce = 1.0;
        } else if ((m_config.guided || (m_config.train_auto && m_online_iteration > 0)) == false ||
                   !has_guidance_distribution) {
            if (has_masked_diffusion_prediction()) {
                if (masked_diffusion_runtime->direction_only_mode()) {
                    // NPM-adapted has no trainable length/type heads. Keep the
                    // discrete proposal fixed and shared instead of silently
                    // treating an absent neural PMF as a degenerate bounce-1 law.
                    sample_bounce = sample_geometric_distr(sampler->next_1d()) + 1;
                    inv_prob_bounce = 1.0 / rb::floor_pdf(
                        (Float) 1e-6f,
                        pdf_geometric_distr(sample_bounce - 1)
                    );
                } else {
                    Float neural_pdf = 0.f;
                    if (sampler->next_1d() < bounce_alpha) {
                        sample_bounce = sample_geometric_distr(sampler->next_1d()) + 1;
                    } else {
                        sample_bounce = masked_diffusion_runtime->sample_bounce(
                            masked_diffusion_prediction,
                            sampler->next_1d(),
                            neural_pdf
                        );
                    }
                    if (sample_bounce <= 0) {
                        sample_bounce = 1;
                    }
                    if (neural_pdf <= 0.f) {
                        neural_pdf = masked_diffusion_runtime->pdf_bounce(
                            masked_diffusion_prediction,
                            sample_bounce
                        );
                    }
                    inv_prob_bounce = 1.0 / rb::floor_pdf(
                        (Float) 1e-6f,
                        length_proposal_mixture_pdf(
                            pdf_geometric_distr(sample_bounce - 1), neural_pdf, bounce_alpha
                        )
                    );
                }
            } else {
                sample_bounce   = sample_geometric_distr(sampler->next_1d()) + 1;
                inv_prob_bounce = 1.0 / pdf_geometric_distr(sample_bounce - 1);
            }
        } else {
            if (sampler->next_1d() < bounce_alpha) {
                sample_bounce = sample_geometric_distr(sampler->next_1d()) + 1;
            } else {
                // Learned discrete bounce indices may be 0; clamp to a valid bounce range.
                sample_bounce = chain_distr->sample_n(sampler, distr_ctx);
                if (sample_bounce <= 0) {
                    sample_bounce = 1;
                }
            }
            inv_prob_bounce = 1.0 / length_proposal_mixture_pdf(
                pdf_geometric_distr(sample_bounce - 1),
                chain_distr->pdf_n(sample_bounce, distr_ctx),
                bounce_alpha
            );
        }

        // For a pure ordered neural proposal, draw the discrete type descriptor
        // once. The initial solve and every Bernoulli re-hit then sample only
        // direction from this cached descriptor state; its exact conditional
        // PMF is applied once inside the inverse-probability estimate.
        prepare_ordered_descriptor(sampler);

        auto ans = specular_manifold_sampling_impl(sampler) * inv_prob_bounce;
        perf_time_all += timer.value();
        return ans;
    }

    static void print_stats() {
        const double t_guide     = (double) perf_time_guide.load()     * 1e-9;
        const double t_bernoulli = (double) perf_time_bernoulli.load() * 1e-9;
        const double t_eval      = (double) perf_time_eval.load()      * 1e-9;
        const double t_all       = (double) perf_time_all.load()       * 1e-9;

        std::cout << "GuidedManifoldSampler stats perf_time_guide " << std::fixed << std::setprecision(6)
                  << t_guide << std::endl;
        std::cout << "GuidedManifoldSampler stats perf_time_bernoulli " << std::fixed << std::setprecision(6)
                  << t_bernoulli << std::endl;
        std::cout << "GuidedManifoldSampler stats perf_time_eval " << std::fixed << std::setprecision(6)
                  << t_eval << std::endl;
        std::cout << "GuidedManifoldSampler stats perf_time_all " << std::fixed << std::setprecision(6)
                  << t_all << std::endl;
        std::cout << "GuidedManifoldSampler diag invalid_bounce " << perf_diag_invalid_bounce << std::endl;
        std::cout << "GuidedManifoldSampler diag next_invalid " << perf_diag_next_invalid << std::endl;
        std::cout << "GuidedManifoldSampler diag next_non_delta " << perf_diag_next_non_delta << std::endl;
        std::cout << "GuidedManifoldSampler diag seed_started " << perf_diag_seed_started << std::endl;
        std::cout << "GuidedManifoldSampler diag chain_invalid " << perf_diag_chain_invalid << std::endl;
        std::cout << "GuidedManifoldSampler diag chain_no_bsdf " << perf_diag_chain_no_bsdf << std::endl;
        std::cout << "GuidedManifoldSampler diag chain_non_delta " << perf_diag_chain_non_delta << std::endl;
        std::cout << "GuidedManifoldSampler diag scatter_fail " << perf_diag_scatter_fail << std::endl;
        std::cout << "GuidedManifoldSampler diag newton_fail " << perf_diag_newton_fail << std::endl;
        std::cout << "GuidedManifoldSampler diag emitter_fail " << perf_diag_emitter_fail << std::endl;
        std::cout << "GuidedManifoldSampler diag emitter_success " << perf_diag_emitter_success << std::endl;
        std::cout << "GuidedManifoldSampler diag bad_tau " << perf_diag_bad_tau << std::endl;
        std::cout << "GuidedManifoldSampler diag bernoulli_fail " << perf_diag_bernoulli_fail << std::endl;
        std::cout << "GuidedManifoldSampler diag bad_offset_pdf " << perf_diag_bad_offset_pdf << std::endl;
        std::cout << "GuidedManifoldSampler diag energy_low " << perf_diag_energy_low << std::endl;
        std::cout << "GuidedManifoldSampler diag stored " << perf_diag_stored << std::endl;
        std::cout << "GuidedManifoldSampler diag candidate_stored " << perf_diag_candidate_stored << std::endl;
        std::cout << "GuidedManifoldSampler diag candidate_limit " << perf_diag_candidate_limit << std::endl;
        std::cout << "GuidedManifoldSampler diag ordered_descriptor_cached "
                  << perf_diag_ordered_descriptor_cached << std::endl;
        std::cout << "GuidedManifoldSampler diag ordered_descriptor_uses "
                  << perf_diag_ordered_descriptor_uses << std::endl;
        std::cout << "GuidedManifoldSampler diag ordered_descriptor_mismatch "
                  << perf_diag_ordered_descriptor_mismatch << std::endl;
    }


    SurfaceInteraction3f si;
    EmitterInteraction ei;
    const Scene *m_scene = nullptr;
    ManifoldPathGuidingConfig m_config;
    std::vector<ManifoldVertex> m_seed_path, m_current_path;
    std::vector<Vector3f> m_offset_normals;
    Manifold_Walk manifold_walk;
    Float m_offset_normals_pdf;
    const std::vector<SubpathSample> *subpath_sample_list     = nullptr;
    const std::vector<SubpathSample> *subpath_sample_list_ext = nullptr;
    std::shared_ptr<SpatialStructure> spatial_structure       = nullptr;
    std::shared_ptr<SpatialStructure> spatial_structure_ext   = nullptr;
    // Runtime fallback joins only when guidance is missing or sparse; it does not replace the primary MPG statistical distribution.
    std::shared_ptr<MaskedDiffusionRuntimeModel> masked_diffusion_runtime = nullptr;
    MaskedDiffusionPrediction masked_diffusion_prediction;
    MaskedDiffusionOrderedDescriptor ordered_descriptor;
    bool ordered_descriptor_cached                            = false;
    int m_timeout             = 0;
    int m_online_iteration                                    = 0;
    bool m_online_last_iteration                              = 0;
    int m_sms_rr_depth                                        = 0;
    int m_sms_max_depth                                       = 0;

    mutable ChainDistributionSamplingContext distr_ctx;

    static inline std::atomic<unsigned long long> perf_time_guide     = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_bernoulli = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_eval      = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_time_all       = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_diag_invalid_bounce = 0;
    static inline std::atomic<unsigned long long> perf_diag_next_invalid = 0;
    static inline std::atomic<unsigned long long> perf_diag_next_non_delta = 0;
    static inline std::atomic<unsigned long long> perf_diag_seed_started = 0;
    static inline std::atomic<unsigned long long> perf_diag_chain_invalid = 0;
    static inline std::atomic<unsigned long long> perf_diag_chain_no_bsdf = 0;
    static inline std::atomic<unsigned long long> perf_diag_chain_non_delta = 0;
    static inline std::atomic<unsigned long long> perf_diag_scatter_fail = 0;
    static inline std::atomic<unsigned long long> perf_diag_newton_fail = 0;
    static inline std::atomic<unsigned long long> perf_diag_emitter_fail = 0;
    static inline std::atomic<unsigned long long> perf_diag_emitter_success = 0;
    static inline std::atomic<unsigned long long> perf_diag_bad_tau = 0;
    static inline std::atomic<unsigned long long> perf_diag_bernoulli_fail = 0;
    static inline std::atomic<unsigned long long> perf_diag_bad_offset_pdf = 0;
    static inline std::atomic<unsigned long long> perf_diag_energy_low = 0;
    static inline std::atomic<unsigned long long> perf_diag_stored = 0;
    static inline std::atomic<unsigned long long> perf_diag_candidate_stored = 0;
    static inline std::atomic<unsigned long long> perf_diag_candidate_limit = 0;
    static inline std::atomic<unsigned long long> perf_diag_ordered_descriptor_cached = 0;
    static inline std::atomic<unsigned long long> perf_diag_ordered_descriptor_uses = 0;
    static inline std::atomic<unsigned long long> perf_diag_ordered_descriptor_mismatch = 0;

    std::shared_ptr<ChainDistribution> chain_distr;
    std::shared_ptr<ChainDistribution> chain_distr_ext;
};

// ===============================
// Integrator
// ===============================
template <typename Float, typename Spectrum>
class ManifoldPathGuidingIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth);
    MTS_IMPORT_TYPES(Scene, Sampler, Sensor, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Medium, ImageBlock);
    using SubpathSample         = ::mitsuba::SubpathSample<Float, Spectrum>;
    using GuidedManifoldSampler = ::mitsuba::GuidedManifoldSampler<Float, Spectrum>;

    using SpatialStructure      = ::mitsuba::SpatialStructure<Float, Spectrum>;
    using SpatialStructureANN   = ::mitsuba::SpatialStructureANN<Float, Spectrum>;
    using SpatialStructureSTree = ::mitsuba::SpatialStructureSTree<Float, Spectrum>;
    using ChainDistribution     = ::mitsuba::ChainDistribution<Float, Spectrum>;
    using MaskedDiffusionRuntimeModel = ::mitsuba::MaskedDiffusionRuntimeModel<Float>;

    using MySTreeNode = ::mitsuba::QuadTreeNode<Float, Spectrum>;

    ManifoldPathGuidingIntegrator(const Properties &props) : Base(props) {
        m_sms_config = ManifoldPathGuidingConfig();

        // core
        m_sms_config.guided     = props.bool_("guided", false); // guide or not
        m_sms_config.train_auto = props.bool_("train_auto", false);

        // consume extra XML props (Mitsuba2 will error if unreferenced)
        const bool debug = props.bool_("debug", false);
        const bool sms_guiding = props.bool_("sms_guiding", m_sms_config.guided);
        (void) debug;
        (void) sms_guiding;

        // initialization
        m_sms_config.initial = props.int_("initial", 0);

        // ablation
        m_sms_config.ablation = props.int_("ablation", 0);

        // spatial struct
        // legacy key (some scenes use "spatial_struct")
        m_sms_config.spatial_struct = props.int_("spatial_struct", 1);

        // keys used by your cbox_mpg.xml
        const int spatial_type = props.int_("spatial_type", -1);
        const int sms_spatial_struct = props.int_("sms_spatial_struct", -1);

        // Prefer the more explicit keys if provided:
        // - if sms_spatial_struct is set, it wins
        // - otherwise spatial_type can override legacy spatial_struct
        if (spatial_type >= 0)
            m_sms_config.spatial_struct = spatial_type;
        if (sms_spatial_struct >= 0)
            m_sms_config.spatial_struct = sms_spatial_struct;

        m_sms_config.spatial_filter = props.float_("spatial_filter", 0.1f);
        m_sms_config.knn_k          = props.int_("knn_k", -1);


        // directional struct
        m_sms_config.directional_struct = props.int_("directional_struct", 0);

        // online training related
        m_sms_config.budget_is_time = props.bool_("budget_is_time", true);
        m_sms_config.train_budget   = props.float_("train_budget", 0.3f);
        m_sms_config.train_fusion   = props.bool_("train_fusion", false);
        m_sms_config.export_samples_path = props.string("export_samples_path", "");
        m_sms_config.export_max_spp_per_round = std::max(1, props.int_("export_max_spp_per_round", 999999));
        m_sms_config.masked_diffusion_runtime_path = props.string("masked_diffusion_runtime_path", "");
        m_sms_config.masked_diffusion_steps = std::max(1, props.int_("diffusion_steps", 1));
        if (!m_sms_config.masked_diffusion_runtime_path.empty()) {
            std::cout << "Masked diffusion runtime diffusion_steps=" << m_sms_config.masked_diffusion_steps << std::endl;
        }
        const float timeout_seconds = props.float_("timeout", -1.f);
        // The first online-training round must read the XML timeout; otherwise it is treated as a zero-second budget and degrades.
        m_timeout = timeout_seconds < 0.f ? -1 : std::max(1, (int) std::ceil(timeout_seconds));

        // utils
        m_sms_config.selective          = props.bool_("selective", false);
        m_sms_config.sms_strict_uniform = props.bool_("sms_strict_uniform", false);
        m_sms_config.sms_bounce_force   = props.int_("sms_bounce_force", 0);
        m_sms_config.sms_type_force     = props.int_("sms_type_force", 0);
        m_sms_config.sms_tau_force      = props.int_("sms_tau_force", -1);
        m_sms_config.seedid             = props.int_("seedid", 2023);
        m_sms_config.rep_ber            = props.int_("rep_ber", 1);
        m_sms_config.bounce_alpha       = props.float_("bounce_alpha", 0.5f);
        if (!is_valid_bounce_alpha(m_sms_config.bounce_alpha)) {
            Throw("bounce_alpha must be finite and in [0, 1]");
        }
        m_sms_config.learned_direction_kappa_scale = props.float_(
            "learned_direction_kappa_scale", 1.f
        );
        if (!std::isfinite((double) m_sms_config.learned_direction_kappa_scale) ||
            m_sms_config.learned_direction_kappa_scale < 0.f) {
            Throw("learned_direction_kappa_scale must be finite and nonnegative");
        }
        m_sms_config.learned_direction_flip_to_receiver_hemisphere = props.bool_(
            "learned_direction_flip_to_receiver_hemisphere", true
        );
        if (!m_sms_config.masked_diffusion_runtime_path.empty()) {
            std::cout << "Masked diffusion runtime learned_direction_kappa_scale="
                      << m_sms_config.learned_direction_kappa_scale
                      << " flip_to_receiver_hemisphere="
                      << m_sms_config.learned_direction_flip_to_receiver_hemisphere
                      << std::endl;
        }
        m_sms_config.learned_proposal_inv_prob_clamp = props.float_(
            "learned_proposal_inv_prob_clamp",
            props.float_("sms_inv_prob_clamp", 0.f)
        );
        m_sms_config.learned_proposal_contribution_clamp = props.float_(
            "learned_proposal_contribution_clamp",
            props.float_("sms_contribution_clamp", 0.f)
        );

        // product sampling
        m_sms_config.product_sampling = props.bool_("product_sampling", false);

        // sms core
        m_sms_config.solver_threshold     = props.float_("solver_threshold", 1e-4f);
        m_sms_config.uniqueness_threshold = props.float_("uniqueness_threshold", 1e-4f);
        m_sms_config.max_trials           = props.int_("max_trials", 1e6);

        // sms legacy
        m_sms_config.halfvector_constraints = props.bool_("halfvector_constraints", false);
        m_sms_config.export_seed_candidates = props.bool_("export_seed_candidates", false);
        m_sms_config.export_seed_candidate_probability =
            props.float_("export_seed_candidate_probability", 0.f);
        m_sms_config.export_seed_candidate_max =
            props.int_("export_seed_candidate_max", 4096);
        m_sms_config.step_scale             = props.float_("step_scale", 1.f);
        m_sms_config.max_iterations         = props.int_("max_iterations", 20);

        // legacy
        m_sms_config.prob_uniform_train = props.float_("prob_uniform_train", 0.5f);
        m_sms_config.prob_uniform       = props.float_("prob_uniform", 0.5f);
        m_sms_config.sms_max_depth      = props.int_("sms_max_depth", 99);

        if (!m_sms_config.masked_diffusion_runtime_path.empty()) {
            masked_diffusion_runtime = std::make_shared<MaskedDiffusionRuntimeModel>();
            if (!masked_diffusion_runtime->load(m_sms_config.masked_diffusion_runtime_path)) {
                masked_diffusion_runtime = nullptr;
                std::cout << "Disable masked diffusion runtime fallback due to load failure." << std::endl;
            }
        }

        global_sms_config = m_sms_config;
    }

    ~ManifoldPathGuidingIntegrator() {
        if (subpath_sample_list) { delete subpath_sample_list; subpath_sample_list = nullptr; }
        if (subpath_sample_list_ext) { delete subpath_sample_list_ext; subpath_sample_list_ext = nullptr; }
    }

    bool render(Scene *scene, Sensor *sensor) override {
        subpath_sample_list     = new std::vector<SubpathSample>();
        subpath_sample_list_ext = new std::vector<SubpathSample>();
        spatial_structure       = spatialStructureFactory<Float, Spectrum>(scene->bbox());
        spatial_structure_ext   = spatialStructureFactory<Float, Spectrum>(scene->bbox());
        // Start each render with an empty training cache so prior samples in-process do not pollute the current guide.
        global_new_data.clear();

        if (m_sms_config.initial == 2) {
            std::cout << "Loading from external recorded samples..." << std::endl;
            std::ifstream in("my_log2.txt");
            std::string line;
            while (std::getline(in, line)) {
                std::stringstream ss(line);
                Point3f xD   = 0;
                Point3f xL   = 0;
                Point3f x1   = 0;
                Float energy = 0;
                int bounce   = 0;
                int tau      = -1;
                ss >> xD[0] >> xD[1] >> xD[2] >>
                    xL[0] >> xL[1] >> xL[2] >>
                    x1[0] >> x1[1] >> x1[2] >>
                    energy >> bounce;
                SubpathSample input_cache;
                input_cache.xD = xD;
                input_cache.xL = xL;
                input_cache.x1 = x1;
                input_cache.energy = energy;
                input_cache.bounce = bounce;
                input_cache.type = tau;
                subpath_sample_list_ext->push_back(input_cache);
            }
            std::cout << "Loaded sample num " << subpath_sample_list_ext->size() << std::endl;
            spatial_structure_ext->build(*subpath_sample_list_ext, m_sms_config.knn_k);
        }

        // Training budget covers online training/rendering only, excluding scene initialization time.
        Timer my_timer;
        bool result;
        if (m_sms_config.train_auto == false) {
            result = MonteCarloIntegrator<Float, Spectrum>::render(scene, sensor);
        } else {
            float total_time = m_timeout;
            if (total_time < 0) {
                total_time = 1e9;
            }
            float total_spp = scene->sensors()[0]->sampler()->m_sample_count;
            int round_id = 0;
            int used_spp = 0;
            // Shifting left by 31 bits is undefined behavior; cap the exponent early.
            auto training_spp_for_round = [](int round) {
                if (round >= 30) {
                    return 1 << 30;
                }
                return 1 << round;
            };

            while (true) {
                std::cout << "Round " << round_id << std::endl;

                float used_time = my_timer.value() * 0.001f;


                // The following code gets information for this iteration

                // Training strategy
                bool is_last_iter  = false;
                bool need_rebuild  = true;
                int this_iter_spp  = 0;
                int this_iter_time = 0;

                if (m_sms_config.budget_is_time) {
                    if (used_time >= total_time * m_sms_config.train_budget) {
                        // Next iteration is rendering
                        need_rebuild   = false;
                        is_last_iter   = true;
                        this_iter_time = total_time - used_time;
                        this_iter_spp  = 999999;
                    } else if (used_time > total_time * m_sms_config.train_budget * 0.5f) {
                        // Next iteration is training (continue this iteration,
                        // do not delete samples)
                        need_rebuild   = false;
                        this_iter_time = total_time * m_sms_config.train_budget - used_time;
                        this_iter_spp  = 999999;
                    } else {
                        // Next iteration is training (new iteration)
                        need_rebuild   = true;
                        this_iter_time = total_time * m_sms_config.train_budget - used_time;
                        this_iter_spp  = training_spp_for_round(round_id);
                    }
                } else {
                    this_iter_time = 1e9;
                    if (used_spp >= total_spp * m_sms_config.train_budget) {
                        // Next iteration is rendering
                        need_rebuild  = false;
                        is_last_iter  = true;
                        this_iter_spp = total_spp - used_spp;
                    } else if (used_spp > total_spp * m_sms_config.train_budget * 0.5f) {
                        // Next iteration is training (continue this iteration,
                        // do not delete samples)
                        need_rebuild  = false;
                        this_iter_spp = total_spp * m_sms_config.train_budget - used_spp;
                    } else {
                        // Next iteration is training (new iteration)
                        need_rebuild  = true;
                        this_iter_spp = training_spp_for_round(round_id);
                    }
                }

                this_iter_time = std::max(1, this_iter_time);
                this_iter_spp  = std::max(1, this_iter_spp);
                const bool export_only_samples = !m_sms_config.export_samples_path.empty();
                if (export_only_samples) {
                    this_iter_spp = std::min(this_iter_spp, m_sms_config.export_max_spp_per_round);
                }

                m_online_iteration                             = round_id;
                g_iter                                         = m_online_iteration;
                m_online_last_iteration                        = is_last_iter;
                scene->sensors()[0]->sampler()->m_sample_count = this_iter_spp;
                m_timeout                                      = this_iter_time;
                used_spp += this_iter_spp;

                if (need_rebuild || is_last_iter) {
                    if (round_id > 0) {
                        // Data export must accumulate samples across rounds; do not stop right after the first valid batch.
                        // Occasional bad training samples can corrupt the spatial structure, so filter strictly first.
                        auto is_finite_point = [](const Point3f &p) {
                            return std::isfinite((double) p[0]) && std::isfinite((double) p[1]) &&
                                   std::isfinite((double) p[2]);
                        };
                        std::vector<SubpathSample> filtered_samples;
                        filtered_samples.reserve(global_new_data.size());
                        size_t invalid_nonfinite = 0;
                        size_t invalid_bounce    = 0;
                        size_t invalid_direction = 0;

                        for (const SubpathSample &sample : global_new_data) {
                            if (sample.bounce <= 0 || sample.bounce >= MAX_CHAIN_LENGTH) {
                                invalid_bounce++;
                                continue;
                            }
                            if (!std::isfinite((double) sample.energy) || !is_finite_point(sample.xD) ||
                                !is_finite_point(sample.xL) || !is_finite_point(sample.x1)) {
                                invalid_nonfinite++;
                                continue;
                            }

                            Vector3f delta = sample.x1 - sample.xD;
                            Float direction_norm2 = dot(delta, delta);
                            if (!std::isfinite((double) direction_norm2) || direction_norm2 <= 1e-12f) {
                                invalid_direction++;
                                continue;
                            }

                            filtered_samples.push_back(sample);
                        }

                        // Multi-threaded block merging order is nondeterministic; sort before rebuild for reproducible online-training order.
                        auto sample_less = [](const SubpathSample &lhs, const SubpathSample &rhs) {
                            auto compare_float = [](Float a, Float b) {
                                if (a < b) {
                                    return -1;
                                }
                                if (a > b) {
                                    return 1;
                                }
                                return 0;
                            };
                            const Point3f *lhs_points[] = { &lhs.xD, &lhs.xL, &lhs.x1 };
                            const Point3f *rhs_points[] = { &rhs.xD, &rhs.xL, &rhs.x1 };
                            for (int point_index = 0; point_index < 3; ++point_index) {
                                for (int axis = 0; axis < 3; ++axis) {
                                    int cmp = compare_float((*lhs_points[point_index])[axis], (*rhs_points[point_index])[axis]);
                                    if (cmp != 0) {
                                        return cmp < 0;
                                    }
                                }
                            }
                            if (lhs.bounce != rhs.bounce) {
                                return lhs.bounce < rhs.bounce;
                            }
                            if (lhs.type != rhs.type) {
                                return lhs.type < rhs.type;
                            }
                            if (lhs.filtered != rhs.filtered) {
                                return lhs.filtered < rhs.filtered;
                            }
                            return lhs.energy < rhs.energy;
                        };
                        std::stable_sort(filtered_samples.begin(), filtered_samples.end(), sample_less);

                        *subpath_sample_list = std::move(filtered_samples);

                        std::cout << "Recorded " << subpath_sample_list->size() << " samples" << std::endl;
                        std::cout << "Filtered invalid samples nonfinite=" << invalid_nonfinite
                                  << " bounce=" << invalid_bounce
                                  << " direction=" << invalid_direction << std::endl;

                        if (export_only_samples) {
                            std::cout << "Export accumulation samples " << subpath_sample_list->size()
                                      << std::endl;
                        } else {
                            const size_t min_guiding_samples = 32;
                            if (subpath_sample_list->size() < min_guiding_samples) {
                                std::cout << "Skip rebuild due to insufficient samples "
                                          << subpath_sample_list->size() << std::endl;
                            } else {
                                if (global_sms_config.spatial_struct != 2) { // 2 means naive sdtree
                                    spatial_structure = spatialStructureFactory<Float, Spectrum>(scene->bbox());
                                }

                                spatial_structure->build(*subpath_sample_list, m_sms_config.knn_k);
                                if (need_rebuild && m_sms_config.train_fusion == false) {
                                    global_new_data.clear();
                                }
                            }
                        }
                    }
                }
                    
                if (m_sms_config.budget_is_time) {
                    const Float remaining=(Float)total_time-(Float)(my_timer.value()*0.001);
                    if (remaining<=0) {
                        std::cout<<"RENDER budget_exhausted_during_learning 1"<<std::endl;
                        break;
                    }
                    // The original derived m_timeout shadows SamplingIntegrator::m_timeout.
                    // Forward the remaining per-round time to the actual render deadline.
                    Base::m_timeout=is_last_iter ? remaining : std::min((Float)this_iter_time,remaining);
                }
                result = MonteCarloIntegrator<Float, Spectrum>::render(scene, sensor);

                if (is_last_iter) {
                    break;
                }

                round_id++;
            }
        }

        if (!m_sms_config.export_samples_path.empty()) {
            export_subpath_samples_jsonl(*subpath_sample_list, m_sms_config.export_samples_path);
        }

        std::cout<<"RENDER total_render_seconds "<<std::fixed<<std::setprecision(6)<<my_timer.value()*0.001<<std::endl;
        spatial_structure->print_stats();
        ChainDistribution::print_stats();
        GuidedManifoldSampler::print_stats();
        print_stats();
        return result;
    }

    void render_block(const Scene *scene, const Sensor *sensor, Sampler *sampler, ImageBlock *block, Float *aovs,
                      size_t sample_count_) const override {
        auto rb_before=rb::lc.paths;
        MonteCarloIntegrator<Float, Spectrum>::render_block(scene, sensor, sampler, block, aovs, sample_count_);
        rb::merge();

        // Record samples for each block
        if (m_sms_config.train_auto &&
            (!m_online_last_iteration || !m_sms_config.export_samples_path.empty())) {
            static std::mutex mu;
            mu.lock();
            std::vector<SubpathSample> &new_data =
                (std::vector<SubpathSample> &) recorded_samples_thread_local<Float, Spectrum>;
            for (const SubpathSample &cache : new_data)
                global_new_data.push_back(cache);
            new_data.clear();
            mu.unlock();
        }
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &ray_,
                                     const Medium * /* medium */, Float * /* aovs */, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);
        rb::lc.paths++; 

        GuidedManifoldSampler &mf = (GuidedManifoldSampler &) thread_mf;
        if constexpr (is_array_v<Float>) {
            Throw("This integrator does not support vector/gpu/autodiff modes!");
            return { 0.f, 0.f };
        } else {
            RayDifferential3f ray = ray_;
            Float eta             = 1.f;
            Spectrum throughput(1.f), result(0.f);
            bool specular_camera_path = true; // To capture emitters visible direcly through purely
            bool saw_delta = false;
            bool still_perform_nee = true;

            // ---------------------- First intersection ----------------------

            SurfaceInteraction3f si = scene->ray_intersect(ray);
            Mask valid_ray          = si.is_valid();
            EmitterPtr emitter      = si.emitter(scene);

            if (emitter) {
                result += emitter->eval(si);
            }

            // ---------------------- Main loop ----------------------

            for (int depth = 1;; ++depth) {

                // ------------------ Possibly terminate path -----------------

                if (!si.is_valid())
                    break;
                si.compute_partials(ray);

                if (depth > m_rr_depth) {
                    Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                    if (sampler->next_1d() > q)
                        break;
                    throughput *= rcp(q);
                }

                if (uint32_t(depth) >= uint32_t(m_max_depth))
                    break;

                // --------------- Specular Manifold Sampling -----------------
                // IMPORTANT: do NOT rely only on shape tags (caustic_caster/receiver).
                // Many reference scenes do not set them, which makes MPG/SMS never trigger.
                // Instead, derive the caster/receiver behavior from the BSDF flags.
                BSDFContext ctx;
                ctx.sampler  = sampler;

                BSDFPtr bsdf = si.bsdf(ray);
                uint32_t bsdf_flags = bsdf ? bsdf->flags() : 0u;

                const bool bsdf_is_delta  = bsdf && has_flag(bsdf_flags, BSDFFlags::Delta);
                const bool bsdf_is_smooth = bsdf && has_flag(bsdf_flags, BSDFFlags::Smooth);

                const bool shape_is_caster =
                    si.shape && (si.shape->is_caustic_caster_multi_scatter() ||
                                 si.shape->is_caustic_caster_single_scatter() ||
                                 si.shape->is_caustic_bouncer());
                const bool shape_is_receiver = si.shape && si.shape->is_caustic_receiver();

                // Caster: prefer explicit scene tags when they exist, while preserving BSDF fallback.
                bool on_caustic_caster = shape_is_caster || bsdf_is_delta;
                saw_delta |= on_caustic_caster;
                if (!on_caustic_caster) {
                    still_perform_nee = true; // this is a new separator, reset flag
                }

                // Receiver-side SMS must also work when a camera ray directly hits a tagged receiver.
                // Water/pool replay export depends on this original SMS behavior.
                bool is_receiver = shape_is_receiver || (bsdf_is_smooth && !bsdf_is_delta);
                bool receiver_side_sms = shape_is_receiver || saw_delta;

                if (receiver_side_sms && is_receiver && !on_caustic_caster && (depth < m_sms_config.sms_max_depth)) {

                    // If the scene has no "caustic emitters" list populated, fall back to all emitters
                    const auto &ce_multi = scene->caustic_emitters_multi_scatter();
                    const auto &ce_single = scene->caustic_emitters_single_scatter();
                    const auto &emitters = (ce_multi.size() > 0) ? ce_multi :
                                           ((ce_single.size() > 0) ? ce_single : scene->emitters());

                    EmitterInteraction ei =
                        SpecularManifold<Float, Spectrum>::sample_emitter_interaction(si, emitters, sampler);

                    mf.m_online_iteration      = m_online_iteration;
                    mf.m_online_last_iteration = m_online_last_iteration;
                    mf.m_sms_max_depth         = std::max(1, m_max_depth - depth);
                    mf.m_sms_rr_depth          = std::max(1, m_rr_depth - depth);

                    mf.init(scene, m_sms_config, si, ei, subpath_sample_list, subpath_sample_list_ext,
                            spatial_structure, spatial_structure_ext, masked_diffusion_runtime);

                    perf_manifold_query++;

                    // Trigger manifold sampling
                    if (!(m_sms_config.guided || m_sms_config.train_auto) || !mf.chain_distr ||
                        (mf.chain_distr && !(mf.chain_distr->weight_not_filtered < 1 &&
                                             m_sms_config.selective && m_online_last_iteration))) {
                        perf_manifold_executed++;
                        still_perform_nee = false;
                        result += throughput * mf.specular_manifold_sampling(sampler);
                    }
                }

                // The following codes are mostly from original SMS
                // --------------------- Emitter sampling ---------------------
                if ((bsdf_is_smooth && !on_caustic_caster) || still_perform_nee) {
                    auto [ds, emitter_weight] = scene->sample_emitter_direction(si, sampler->next_2d(), true);
                    if (ds.pdf != 0.f) {
                        Vector3f wo       = si.to_local(ds.d);
                        Spectrum bsdf_val = bsdf->eval(ctx, si, wo);
                        bsdf_val          = si.to_world_mueller(bsdf_val, -wo, si.wi);
                        Float bsdf_pdf = bsdf->pdf(ctx, si, wo);
                        Float mis      = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                        result += mis * throughput * bsdf_val * emitter_weight;
                    }
                }

                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_weight] = bsdf->sample(ctx, si, sampler->next_1d(), sampler->next_2d());
                bsdf_weight            = si.to_world_mueller(bsdf_weight, -bs.wo, si.wi);

                throughput = throughput * bsdf_weight;
                eta *= bs.eta;
                if (!has_flag(bs.sampled_type, BSDFFlags::Delta)) {
                    specular_camera_path = false;
                }
                if (all(eq(throughput, 0.f)))
                    break;

                ray                          = si.spawn_ray(si.to_world(bs.wo));
                SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray);
                emitter                      = si_bsdf.emitter(scene);

                // Hit emitter after BSDF sampling
                if (emitter) {
                    if (!on_caustic_caster || (!specular_camera_path && still_perform_nee) ||
                        emitter->is_environment()) { // filter out glints
                        Spectrum emitter_val = emitter->eval(si_bsdf);
                        DirectionSample3f ds(si_bsdf, si);
                        ds.object         = emitter;
                        Float emitter_pdf = select(!has_flag(bs.sampled_type, BSDFFlags::Delta),
                                                   scene->pdf_emitter_direction(si, ds), 0.f);
                        Float mis         = mis_weight(bs.pdf, emitter_pdf);
                        result += mis * throughput * emitter_val;
                    }
                }
                si = std::move(si_bsdf);
            }

            return { result, valid_ray };
        }
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    void print_stats() {
        const unsigned long long q = perf_manifold_query.load();
        const unsigned long long e = perf_manifold_executed.load();

        std::cout << "Manifold perf_manifold_query " << q << std::endl;
        std::cout << "Manifold perf_manifold_executed " << e << std::endl;

        float ratio = 0.f;
        if (q > 0)
            ratio = (float) e / (float) q;

        std::cout << "Manifold perf_manifold_executed_ratio " << ratio << std::endl;
    }

    MTS_DECLARE_CLASS()

public:
    ManifoldPathGuidingConfig m_sms_config;
    std::vector<SubpathSample> *subpath_sample_list = nullptr;
    std::vector<SubpathSample> *subpath_sample_list_ext = nullptr;
    std::shared_ptr<SpatialStructure> spatial_structure;
    std::shared_ptr<SpatialStructure> spatial_structure_ext;
    std::shared_ptr<MaskedDiffusionRuntimeModel> masked_diffusion_runtime;
    mutable std::vector<SubpathSample> global_new_data;
    static inline ThreadLocal<GuidedManifoldSampler> thread_mf;
    int m_timeout             = 0;
    int m_online_iteration       = 0;
    bool m_online_last_iteration = 0;

    static inline std::atomic<unsigned long long> perf_manifold_executed = 0; // nano sec
    static inline std::atomic<unsigned long long> perf_manifold_query    = 0; // nano sec
};

MTS_IMPLEMENT_CLASS_VARIANT(ManifoldPathGuidingIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ManifoldPathGuidingIntegrator, "manifold path guiding integrator");
NAMESPACE_END(mitsuba)
