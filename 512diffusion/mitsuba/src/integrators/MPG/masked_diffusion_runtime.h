#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "util.h"

NAMESPACE_BEGIN(mitsuba)

template <typename Float> class MaskedDiffusionRuntimeModel {
public:
    using Scalar = scalar_t<Float>;
    using Vector3f = Vector<Scalar, 3>;

    struct Prediction {
        bool valid = false;
        bool ordered_context_valid = false;
        bool direction_only = false;
        std::vector<Scalar> chain_length_logits;
        std::vector<Scalar> chain_length_probs;
        std::vector<Scalar> chain_type_logits;
        Vector3f direction = Vector3f(0.f);
        std::vector<Scalar> vmf_weights;
        std::vector<Vector3f> vmf_means;
        std::vector<Scalar> vmf_kappas;
        std::vector<Scalar> ordered_condition;
    };

    /// One complete ordered descriptor draw together with its exact discrete PMF.
    struct OrderedDescriptorSample {
        bool valid = false;
        int chain_length = 1;
        std::vector<int> chain_type_bits;
        int tau = 0;
        Scalar length_probability = 0.f;
        std::vector<Scalar> transition_probabilities;
        Scalar conditional_log_probability = 0.f;
        Scalar conditional_probability = 1.f;
        Scalar descriptor_log_probability = -std::numeric_limits<Scalar>::infinity();
        Scalar descriptor_probability = 0.f;
        Prediction direction_prediction;
    };

    bool load(const std::string &path) {
        // Runtime blobs are ASCII so humans can grep/diff checkpoints quickly.
        if (path.empty()) {
            return false;
        }

        std::ifstream in(path);
        if (!in.is_open()) {
            std::cout << "Failed to open masked diffusion runtime file: " << path << std::endl;
            return false;
        }

        std::string magic;
        in >> magic;
        if (magic != "masked_diffusion_runtime_v1") {
            std::cout << "Unexpected masked diffusion runtime magic: " << magic << std::endl;
            return false;
        }

        clear();

        std::string token;
        while (in >> token) {
            if (token == "hidden_dim") {
                in >> hidden_dim;
            } else if (token == "depth") {
                in >> depth;
            } else if (token == "condition_dim") {
                in >> condition_dim;
            } else if (token == "backbone_type") {
                in >> backbone_type;
            } else if (token == "direction_distribution") {
                in >> direction_distribution;
            } else if (token == "vmf_components") {
                in >> vmf_components;
            } else if (token == "vmf_min_kappa") {
                in >> vmf_min_kappa;
            } else if (token == "vmf_max_kappa") {
                in >> vmf_max_kappa;
            } else if (token == "transformer_heads") {
                in >> transformer_heads;
            } else if (token == "transformer_feedforward_dim") {
                in >> transformer_feedforward_dim;
            } else if (token == "token_count") {
                in >> token_count;
            } else if (token == "time_embedding_dim") {
                in >> time_embedding_dim;
            } else if (token == "token_embedding_dim") {
                in >> token_embedding_dim;
            } else if (token == "runtime_input_mode") {
                in >> runtime_input_mode;
            } else if (token == "discrete_diffusion_mode") {
                in >> discrete_diffusion_mode;
            } else if (token == "chain_length_classes") {
                in >> chain_length_classes;
            } else if (token == "chain_type_bits") {
                in >> chain_type_bits;
            } else if (token == "layer_norm_eps") {
                in >> layer_norm_eps;
            } else if (token == "npm_grid_levels") {
                in >> npm_grid_levels;
            } else if (token == "npm_grid_features") {
                in >> npm_grid_features;
            } else if (token == "npm_sh_degree") {
                in >> npm_sh_degree;
            } else if (token == "npm_roughness_default") {
                in >> npm_roughness_default;
            } else if (token == "npm_direction_source") {
                in >> npm_direction_source;
            } else if (token == "npm_descriptor_policy_id") {
                in >> npm_descriptor_policy_id;
            } else if (token == "tensor") {
                std::string name;
                size_t ndim = 0;
                in >> name >> ndim;
                std::vector<size_t> dims(ndim);
                size_t value_count = 1;
                for (size_t i = 0; i < ndim; ++i) {
                    in >> dims[i];
                    value_count *= dims[i];
                }

                TensorData tensor;
                tensor.dims = dims;
                tensor.values.resize(value_count);
                for (size_t i = 0; i < value_count; ++i) {
                    in >> tensor.values[i];
                }
                tensors[name] = std::move(tensor);
            } else {
                std::cout << "Unknown token in masked diffusion runtime file: " << token << std::endl;
                return false;
            }
        }

        loaded = validate();
        if (loaded) {
            std::cout << "Loaded masked diffusion runtime model from " << path
                      << " (discrete_diffusion_mode=" << discrete_diffusion_mode
                      << ", runtime_input_mode=" << runtime_input_mode << ")"
                      << std::endl;
        }
        return loaded;
    }

    bool valid() const { return loaded; }

    Prediction predict(const std::vector<Scalar> &condition) const {
        return predict(condition, 1);
    }

    Prediction predict(const std::vector<Scalar> &condition, int diffusion_steps) const {
        Prediction prediction;
        if (!loaded) {
            return prediction;
        }
        if ((int) condition.size() != condition_dim) {
            return prediction;
        }

        diffusion_steps = std::max(1, diffusion_steps);

        if (backbone_type == "npm_adapted") {
            return predict_npm_adapted_from_condition(condition);
        }

        // Ordered absorbing generation is a joint descriptor sampler. Keep
        // this result as a context: the integrator uses its length PMF in the
        // existing mixture, then samples type bits conditional on the selected
        // length through sample_ordered_descriptor_for_length().
        if (discrete_diffusion_mode == "ordered_absorbing") {
            return prepare_ordered_context(condition);
        }

        // Full MLP runtimes retain timestep, noisy direction, and mask-token
        // embeddings. They can therefore reproduce the Phase11 autoregressive
        // loop. Legacy folded/condition-only runtimes fall back to one forward.
        if (backbone_type == "mlp" && runtime_input_mode == "full_mlp") {
            return predict_mlp_multistep(condition, diffusion_steps);
        }

        return predict_single_step(condition);
    }

    bool ordered_descriptor_mode() const {
        return loaded && discrete_diffusion_mode == "ordered_absorbing";
    }

    bool direction_only_mode() const {
        return loaded && backbone_type == "npm_adapted";
    }

    Prediction predict_npm_adapted(
        const Vector3f &position,
        const Vector3f &outgoing_direction,
        const Vector3f &normal,
        Scalar roughness
    ) const {
        if (!direction_only_mode() || !std::isfinite((double) roughness)) {
            return Prediction();
        }
        return run_npm_adapted(
            position,
            outgoing_direction,
            normal,
            std::min(std::max(roughness, (Scalar) 0.f), (Scalar) 1.f)
        );
    }

    OrderedDescriptorSample sample_ordered_descriptor(
        const Prediction &context,
        Scalar u_length,
        const std::vector<Scalar> &u_bits
    ) const {
        OrderedDescriptorSample sample;
        if (!ordered_descriptor_mode() || !context.ordered_context_valid ||
            (int) context.ordered_condition.size() != condition_dim ||
            !is_unit_sample(u_length)) {
            return sample;
        }

        Scalar length_probability = 0.f;
        const int chain_length = sample_positive_chain_length(
            context.chain_length_probs,
            u_length,
            length_probability
        );
        if (chain_length <= 0 || length_probability <= (Scalar) 0.f) {
            return sample;
        }

        return sample_ordered_descriptor_for_length(
            context,
            chain_length,
            u_bits
        );
    }

    OrderedDescriptorSample sample_ordered_descriptor_for_length(
        const Prediction &context,
        int chain_length,
        const std::vector<Scalar> &u_bits
    ) const {
        OrderedDescriptorSample sample;
        if (!ordered_descriptor_mode() || !context.ordered_context_valid ||
            (int) context.ordered_condition.size() != condition_dim ||
            chain_length <= 0 || chain_length > chain_type_bits ||
            chain_length >= (int) context.chain_length_probs.size() ||
            (int) u_bits.size() < chain_length) {
            return sample;
        }

        const Scalar length_probability = positive_chain_length_probability(
            context.chain_length_probs,
            chain_length
        );

        std::vector<int> bits(chain_type_bits, 0);
        for (int bit = 0; bit < chain_length; ++bit) {
            bits[bit] = 2;
        }

        sample.chain_length = chain_length;
        sample.chain_type_bits.assign(bits.begin(), bits.begin() + chain_length);
        sample.length_probability = length_probability;
        sample.transition_probabilities.reserve((size_t) chain_length);
        sample.conditional_log_probability = 0.f;
        sample.descriptor_log_probability = length_probability > (Scalar) 0.f
            ? safe_log_probability(length_probability)
            : -std::numeric_limits<Scalar>::infinity();

        const std::vector<Scalar> zero_direction(3, (Scalar) 0.f);
        for (int bit = 0; bit < chain_length; ++bit) {
            const Scalar timestep = (Scalar) (chain_length - bit) / (Scalar) chain_length;
            Prediction transition = predict_diffusion_state(
                context.ordered_condition,
                zero_direction,
                chain_length,
                bits,
                timestep
            );
            if (!transition.valid || bit >= (int) transition.chain_type_logits.size()) {
                return OrderedDescriptorSample();
            }

            const Scalar probability_one = ordered_bit_probability_one(
                transition.chain_type_logits[bit]
            );
            if (!is_unit_sample(u_bits[bit])) {
                return OrderedDescriptorSample();
            }
            const Scalar uniform = clamp_unit_sample(u_bits[bit]);
            const int value = uniform < probability_one ? 1 : 0;
            const Scalar selected_probability = value != 0
                ? probability_one
                : ((Scalar) 1.f - probability_one);
            bits[bit] = value;
            sample.chain_type_bits[bit] = value;
            sample.transition_probabilities.push_back(selected_probability);
            const Scalar transition_log_probability =
                safe_log_probability(selected_probability);
            sample.conditional_log_probability += transition_log_probability;
            sample.descriptor_log_probability += transition_log_probability;
        }

        sample.direction_prediction = predict_diffusion_state(
            context.ordered_condition,
            zero_direction,
            chain_length,
            bits,
            (Scalar) 0.f
        );
        if (!sample.direction_prediction.valid) {
            return OrderedDescriptorSample();
        }

        sample.tau = 0;
        set_bit(sample.tau, chain_length);
        for (int bit = 0; bit < chain_length; ++bit) {
            set_chaintype_bit(sample.tau, bit, bits[bit]);
        }
        sample.conditional_probability = std::exp(
            sample.conditional_log_probability
        );
        sample.descriptor_probability = std::exp(sample.descriptor_log_probability);
        sample.valid = true;
        return sample;
    }

    Scalar log_pdf_ordered_descriptor(
        const Prediction &context,
        int chain_length,
        const std::vector<int> &chain_bits
    ) const {
        if (!ordered_descriptor_mode() || !context.ordered_context_valid ||
            (int) context.ordered_condition.size() != condition_dim ||
            chain_length <= 0 || chain_length > chain_type_bits ||
            chain_length >= (int) context.chain_length_probs.size() ||
            (int) chain_bits.size() < chain_length) {
            return -std::numeric_limits<Scalar>::infinity();
        }

        const Scalar length_probability = positive_chain_length_probability(
            context.chain_length_probs,
            chain_length
        );
        if (length_probability <= (Scalar) 0.f) {
            return -std::numeric_limits<Scalar>::infinity();
        }

        Scalar log_probability = safe_log_probability(length_probability);
        std::vector<int> bits(chain_type_bits, 0);
        for (int bit = 0; bit < chain_length; ++bit) {
            bits[bit] = 2;
        }
        const std::vector<Scalar> zero_direction(3, (Scalar) 0.f);
        for (int bit = 0; bit < chain_length; ++bit) {
            if (chain_bits[bit] != 0 && chain_bits[bit] != 1) {
                return -std::numeric_limits<Scalar>::infinity();
            }
            Prediction transition = predict_diffusion_state(
                context.ordered_condition,
                zero_direction,
                chain_length,
                bits,
                (Scalar) (chain_length - bit) / (Scalar) chain_length
            );
            if (!transition.valid || bit >= (int) transition.chain_type_logits.size()) {
                return -std::numeric_limits<Scalar>::infinity();
            }
            const int value = chain_bits[bit];
            const Scalar probability_one = ordered_bit_probability_one(
                transition.chain_type_logits[bit]
            );
            const Scalar selected_probability = value != 0
                ? probability_one
                : ((Scalar) 1.f - probability_one);
            log_probability += safe_log_probability(selected_probability);
            bits[bit] = value;
        }
        return log_probability;
    }

    Scalar pdf_ordered_descriptor(
        const Prediction &context,
        int chain_length,
        const std::vector<int> &chain_bits
    ) const {
        const Scalar log_probability = log_pdf_ordered_descriptor(
            context,
            chain_length,
            chain_bits
        );
        return std::isfinite((double) log_probability)
            ? std::exp(log_probability)
            : (Scalar) 0.f;
    }

    int sample_bounce(const Prediction &prediction, Scalar sample, Scalar &pdf) const {
        pdf = 0.f;
        if ((!prediction.valid && !prediction.ordered_context_valid) ||
            prediction.chain_length_probs.empty()) {
            return 1;
        }

        Scalar valid_mass = valid_chain_length_mass(prediction);
        if (valid_mass <= (Scalar) 1e-6f) {
            pdf = 1.f;
            return 1;
        }

        Scalar target = std::min(std::max(sample, (Scalar) 0.f), (Scalar) 1.f) * valid_mass;
        Scalar accum = 0.f;
        // Replay marks chain-length bucket 0 invalid—drop it mass then re-normalize positives.
        for (size_t index = 1; index < prediction.chain_length_probs.size(); ++index) {
            accum += prediction.chain_length_probs[index];
            if (target <= accum || index + 1 == prediction.chain_length_probs.size()) {
                pdf = std::max((Scalar) 1e-6f, prediction.chain_length_probs[index] / valid_mass);
                return std::max(1, (int) index);
            }
        }

        pdf = 1.f;
        return 1;
    }

    Scalar pdf_bounce(const Prediction &prediction, int bounce) const {
        if ((!prediction.valid && !prediction.ordered_context_valid) ||
            bounce < 0 || bounce >= (int) prediction.chain_length_probs.size()) {
            return 0.f;
        }
        if (bounce == 0) {
            return 0.f;
        }
        Scalar valid_mass = valid_chain_length_mass(prediction);
        if (valid_mass <= (Scalar) 1e-6f) {
            return 0.f;
        }
        return prediction.chain_length_probs[bounce] / valid_mass;
    }

    int predict_tau(const Prediction &prediction, int bounce) const {
        if (prediction.direction_only || direction_only_mode()) {
            return -1;
        }
        int clamped_bounce = std::max(1, std::min(bounce, chain_type_bits));
        int tau = 0;
        set_bit(tau, clamped_bounce);
        for (int index = 0; index < clamped_bounce; ++index) {
            if (index >= (int) prediction.chain_type_logits.size()) {
                return tau;
            }
            Scalar logit = prediction.chain_type_logits[index];
            set_chaintype_bit(tau, index, logit >= 0.f ? 1 : 0);
        }
        return tau;
    }

    Vector3f sample_direction(const Prediction &prediction, Scalar u_component, Scalar u1, Scalar u2) const {
        return sample_direction(
            prediction, u_component, u1, u2, (Scalar) 1.f
        );
    }

    Vector3f sample_direction(const Prediction &prediction, Scalar u_component, Scalar u1,
                              Scalar u2, Scalar kappa_scale) const {
        validate_direction_kappa_scale(kappa_scale);
        if (!prediction.valid || prediction.vmf_weights.empty()) {
            return prediction.direction;
        }

        Scalar target = std::min(std::max(u_component, (Scalar) 0.f), (Scalar) 1.f);
        Scalar accum = 0.f;
        size_t component = prediction.vmf_weights.size() - 1;
        for (size_t index = 0; index < prediction.vmf_weights.size(); ++index) {
            accum += prediction.vmf_weights[index];
            if (target <= accum) {
                component = index;
                break;
            }
        }
        return sample_vmf(
            prediction.vmf_means[component],
            scaled_vmf_kappa(prediction.vmf_kappas[component], kappa_scale),
            u1,
            u2
        );
    }

    Scalar pdf_direction(const Prediction &prediction, const Vector3f &direction) const {
        return pdf_direction(prediction, direction, (Scalar) 1.f);
    }

    Scalar pdf_direction(const Prediction &prediction, const Vector3f &direction,
                         Scalar kappa_scale) const {
        validate_direction_kappa_scale(kappa_scale);
        if (!prediction.valid || prediction.vmf_weights.empty()) {
            return 0.f;
        }

        Vector3f normalized = normalize_vector3(direction);
        Scalar density = 0.f;
        for (size_t index = 0; index < prediction.vmf_weights.size(); ++index) {
            Scalar kappa = scaled_vmf_kappa(
                prediction.vmf_kappas[index], kappa_scale
            );
            if (kappa < (Scalar) 1e-5f) {
                density += prediction.vmf_weights[index] /
                           ((Scalar) 4.f * math::Pi<Scalar>);
                continue;
            }
            Scalar log_c = log_vmf_normalizer(kappa);
            Scalar cosine = std::min(
                std::max(dot(prediction.vmf_means[index], normalized), (Scalar) -1.f),
                (Scalar) 1.f
            );
            density += prediction.vmf_weights[index] * std::exp(log_c + kappa * cosine);
        }
        return density;
    }

private:
    struct TensorData {
        std::vector<size_t> dims;
        std::vector<Scalar> values;
    };

    void validate_direction_kappa_scale(Scalar kappa_scale) const {
        if (!std::isfinite((double) kappa_scale) || kappa_scale < (Scalar) 0.f) {
            throw std::invalid_argument(
                "learned direction kappa scale must be finite and nonnegative"
            );
        }
    }

    Scalar scaled_vmf_kappa(Scalar kappa, Scalar kappa_scale) const {
        const double scaled = (double) kappa * (double) kappa_scale;
        if (scaled >= (double) std::numeric_limits<Scalar>::max()) {
            return std::numeric_limits<Scalar>::max();
        }
        return std::max((Scalar) scaled, (Scalar) 0.f);
    }

    void clear() {
        loaded = false;
        backbone_type = "mlp";
        direction_distribution = "vector";
        hidden_dim = 0;
        depth = 0;
        condition_dim = 0;
        vmf_components = 0;
        vmf_min_kappa = 1e-4f;
        vmf_max_kappa = 128.f;
        transformer_heads = 0;
        transformer_feedforward_dim = 0;
        token_count = 0;
        time_embedding_dim = 0;
        token_embedding_dim = 0;
        runtime_input_mode = "condition_only";
        discrete_diffusion_mode = "legacy";
        chain_length_classes = 0;
        chain_type_bits = 0;
        layer_norm_eps = (Scalar) 1e-5f;
        npm_grid_levels = 0;
        npm_grid_features = 0;
        npm_sh_degree = 0;
        npm_roughness_default = (Scalar) 0.5f;
        npm_direction_source = "condition_delta_proxy";
        npm_descriptor_policy_id.clear();
        tensors.clear();
    }

    bool validate() const {
        if (hidden_dim <= 0 || depth <= 0 || condition_dim <= 0) {
            return false;
        }
        if (backbone_type == "npm_adapted") {
            return validate_npm_adapted();
        }
        if (chain_length_classes <= 0 || chain_type_bits <= 0) {
            return false;
        }
        if (discrete_diffusion_mode != "legacy" &&
            discrete_diffusion_mode != "ordered_absorbing") {
            return false;
        }

        if (!validate_tensor("chain_length_head_weight", { (size_t) chain_length_classes, (size_t) hidden_dim }) ||
            !validate_tensor("chain_length_head_bias", { (size_t) chain_length_classes })) {
            return false;
        }

        if (direction_distribution == "vmf_mixture") {
            if (vmf_components <= 0 || !std::isfinite(vmf_min_kappa) ||
                !std::isfinite(vmf_max_kappa) || vmf_min_kappa <= 0.f ||
                vmf_max_kappa <= 0.f ||
                !validate_tensor("vmf_logits_head_weight", { (size_t) vmf_components, (size_t) hidden_dim }) ||
                !validate_tensor("vmf_logits_head_bias", { (size_t) vmf_components }) ||
                !validate_tensor("vmf_mean_head_weight", { (size_t) (vmf_components * 3), (size_t) hidden_dim }) ||
                !validate_tensor("vmf_mean_head_bias", { (size_t) (vmf_components * 3) }) ||
                !validate_tensor("vmf_kappa_head_weight", { (size_t) vmf_components, (size_t) hidden_dim }) ||
                !validate_tensor("vmf_kappa_head_bias", { (size_t) vmf_components })) {
                return false;
            }
        } else if (!validate_tensor("direction_head_weight", { 3u, (size_t) hidden_dim }) ||
                   !validate_tensor("direction_head_bias", { 3u })) {
            return false;
        }

        if (backbone_type == "transformer") {
            const int feedforward_dim = transformer_feedforward_dim > 0
                ? transformer_feedforward_dim
                : hidden_dim * 4;
            const bool full_transformer_runtime = runtime_input_mode == "full_transformer";
            if (token_count <= 0 || transformer_heads <= 0 || hidden_dim % transformer_heads != 0 ||
                !validate_tensor("final_norm_weight", { (size_t) hidden_dim }) ||
                !validate_tensor("final_norm_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("condition_token_weight", { (size_t) hidden_dim, (size_t) condition_dim }) ||
                !validate_tensor("condition_token_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("transformer_position_embedding", { 1u, (size_t) token_count, (size_t) hidden_dim }) ||
                !validate_tensor("chain_type_token_head_weight", { 1u, (size_t) hidden_dim }) ||
                !validate_tensor("chain_type_token_head_bias", { 1u })) {
                return false;
            }
            if (full_transformer_runtime) {
                if (time_embedding_dim <= 0 || token_embedding_dim <= 0 ||
                    !validate_tensor("direction_time_token_weight", { (size_t) hidden_dim, (size_t) (3 + time_embedding_dim) }) ||
                    !validate_tensor("direction_time_token_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor("chain_length_token_weight", { (size_t) hidden_dim, (size_t) token_embedding_dim }) ||
                    !validate_tensor("chain_length_token_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor("chain_type_token_weight", { (size_t) hidden_dim, (size_t) token_embedding_dim }) ||
                    !validate_tensor("chain_type_token_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor("chain_length_embedding_weight", { (size_t) (chain_length_classes + 1), (size_t) token_embedding_dim }) ||
                    !validate_tensor("chain_type_embedding_weight", { 3u, (size_t) token_embedding_dim })) {
                    return false;
                }
            } else if (!validate_tensor(
                           "runtime_token_constants",
                           { (size_t) (token_count - 1), (size_t) hidden_dim }
                       )) {
                return false;
            }
            for (int block_index = 0; block_index < depth; ++block_index) {
                const std::string prefix = "transformer_" + std::to_string(block_index);
                if (!validate_tensor(prefix + "_self_attn_in_proj_weight", { (size_t) (3 * hidden_dim), (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_self_attn_in_proj_bias", { (size_t) (3 * hidden_dim) }) ||
                    !validate_tensor(prefix + "_self_attn_out_proj_weight", { (size_t) hidden_dim, (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_self_attn_out_proj_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_linear1_weight", { (size_t) feedforward_dim, (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_linear1_bias", { (size_t) feedforward_dim }) ||
                    !validate_tensor(prefix + "_linear2_weight", { (size_t) hidden_dim, (size_t) feedforward_dim }) ||
                    !validate_tensor(prefix + "_linear2_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_norm1_weight", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_norm1_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_norm2_weight", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_norm2_bias", { (size_t) hidden_dim })) {
                    return false;
                }
            }
        } else if (backbone_type == "dit") {
            const int feedforward_dim = transformer_feedforward_dim > 0
                ? transformer_feedforward_dim
                : hidden_dim * 4;
            if (runtime_input_mode != "full_dit" ||
                token_count != 2 + chain_type_bits ||
                transformer_heads <= 0 || hidden_dim % transformer_heads != 0 ||
                time_embedding_dim <= 0 || token_embedding_dim <= 0 ||
                !validate_tensor("dit_condition_weight", { (size_t) hidden_dim, (size_t) condition_dim }) ||
                !validate_tensor("dit_condition_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("dit_time_linear1_weight", { (size_t) hidden_dim, (size_t) time_embedding_dim }) ||
                !validate_tensor("dit_time_linear1_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("dit_time_linear2_weight", { (size_t) hidden_dim, (size_t) hidden_dim }) ||
                !validate_tensor("dit_time_linear2_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("dit_direction_token_weight", { (size_t) hidden_dim, 3u }) ||
                !validate_tensor("dit_direction_token_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("chain_length_token_weight", { (size_t) hidden_dim, (size_t) token_embedding_dim }) ||
                !validate_tensor("chain_length_token_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("chain_type_token_weight", { (size_t) hidden_dim, (size_t) token_embedding_dim }) ||
                !validate_tensor("chain_type_token_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("chain_length_embedding_weight", { (size_t) (chain_length_classes + 1), (size_t) token_embedding_dim }) ||
                !validate_tensor("chain_type_embedding_weight", { 3u, (size_t) token_embedding_dim }) ||
                !validate_tensor("dit_position_embedding", { 1u, (size_t) token_count, (size_t) hidden_dim }) ||
                !validate_tensor("dit_final_adaln_weight", { (size_t) (2 * hidden_dim), (size_t) hidden_dim }) ||
                !validate_tensor("dit_final_adaln_bias", { (size_t) (2 * hidden_dim) }) ||
                !validate_tensor("chain_type_token_head_weight", { 1u, (size_t) hidden_dim }) ||
                !validate_tensor("chain_type_token_head_bias", { 1u })) {
                return false;
            }
            for (int block_index = 0; block_index < depth; ++block_index) {
                const std::string prefix = "dit_" + std::to_string(block_index);
                if (!validate_tensor(prefix + "_self_attn_in_proj_weight", { (size_t) (3 * hidden_dim), (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_self_attn_in_proj_bias", { (size_t) (3 * hidden_dim) }) ||
                    !validate_tensor(prefix + "_self_attn_out_proj_weight", { (size_t) hidden_dim, (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_self_attn_out_proj_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_linear1_weight", { (size_t) feedforward_dim, (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_linear1_bias", { (size_t) feedforward_dim }) ||
                    !validate_tensor(prefix + "_linear2_weight", { (size_t) hidden_dim, (size_t) feedforward_dim }) ||
                    !validate_tensor(prefix + "_linear2_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_adaln_weight", { (size_t) (6 * hidden_dim), (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_adaln_bias", { (size_t) (6 * hidden_dim) })) {
                    return false;
                }
            }
        } else if (backbone_type == "mlp") {
            const bool full_mlp_runtime = runtime_input_mode == "full_mlp";
            const size_t expected_input_dim = full_mlp_runtime
                ? (size_t) (condition_dim + 3 + time_embedding_dim + token_embedding_dim + token_embedding_dim * chain_type_bits)
                : (size_t) condition_dim;
            if (!validate_tensor("final_norm_weight", { (size_t) hidden_dim }) ||
                !validate_tensor("final_norm_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("input_weight", { (size_t) hidden_dim, expected_input_dim }) ||
                !validate_tensor("input_bias", { (size_t) hidden_dim }) ||
                !validate_tensor("chain_type_head_weight", { (size_t) chain_type_bits, (size_t) hidden_dim }) ||
                !validate_tensor("chain_type_head_bias", { (size_t) chain_type_bits })) {
                return false;
            }
            if (full_mlp_runtime) {
                if (time_embedding_dim <= 0 || token_embedding_dim <= 0 ||
                    !validate_tensor("chain_length_embedding_weight", { (size_t) (chain_length_classes + 1), (size_t) token_embedding_dim }) ||
                    !validate_tensor("chain_type_embedding_weight", { 3u, (size_t) token_embedding_dim })) {
                    return false;
                }
            }
            for (int block_index = 0; block_index < depth; ++block_index) {
                const std::string prefix = "block_" + std::to_string(block_index);
                if (!validate_tensor(prefix + "_norm_weight", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_norm_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_linear1_weight", { (size_t) hidden_dim, (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_linear1_bias", { (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_linear2_weight", { (size_t) hidden_dim, (size_t) hidden_dim }) ||
                    !validate_tensor(prefix + "_linear2_bias", { (size_t) hidden_dim })) {
                    return false;
                }
            }
        } else if (backbone_type == "plain_mlp") {
            if (runtime_input_mode != "condition_only" ||
                !validate_tensor("chain_type_head_weight", { (size_t) chain_type_bits, (size_t) hidden_dim }) ||
                !validate_tensor("chain_type_head_bias", { (size_t) chain_type_bits })) {
                return false;
            }

            for (int layer_index = 0; layer_index < depth; ++layer_index) {
                const std::string prefix = "plain_layer_" + std::to_string(layer_index);
                const size_t input_dim = layer_index == 0
                    ? (size_t) condition_dim
                    : (size_t) hidden_dim;
                if (!validate_tensor(prefix + "_weight", { (size_t) hidden_dim, input_dim }) ||
                    !validate_tensor(prefix + "_bias", { (size_t) hidden_dim })) {
                    return false;
                }
            }
        } else {
            return false;
        }

        if (discrete_diffusion_mode == "ordered_absorbing") {
            const bool dynamic_mlp =
                backbone_type == "mlp" && runtime_input_mode == "full_mlp";
            const bool dynamic_transformer =
                backbone_type == "transformer" && runtime_input_mode == "full_transformer";
            const bool dynamic_dit =
                backbone_type == "dit" && runtime_input_mode == "full_dit";
            if (!dynamic_mlp && !dynamic_transformer && !dynamic_dit) {
                return false;
            }
        }

        return true;
    }

    bool validate_npm_adapted() const {
        if (direction_distribution != "npm_vmf_mixture" ||
            runtime_input_mode != "npm_geometry_proxy" ||
            discrete_diffusion_mode != "legacy" ||
            vmf_components <= 0 || npm_grid_levels <= 0 ||
            npm_grid_features <= 0 || npm_sh_degree != 2 ||
            !std::isfinite((double) vmf_min_kappa) ||
            !std::isfinite((double) vmf_max_kappa) ||
            !std::isfinite((double) npm_roughness_default) ||
            vmf_min_kappa <= (Scalar) 0.f ||
            vmf_max_kappa < vmf_min_kappa ||
            npm_roughness_default < (Scalar) 0.f ||
            npm_roughness_default > (Scalar) 1.f ||
            npm_direction_source != "condition_delta_proxy" ||
            npm_descriptor_policy_id.empty() ||
            !validate_tensor("npm_position_min", { 3u }) ||
            !validate_tensor("npm_position_max", { 3u })) {
            return false;
        }

        const TensorData &position_min = tensor("npm_position_min");
        const TensorData &position_max = tensor("npm_position_max");
        for (size_t axis = 0; axis < 3; ++axis) {
            if (!std::isfinite((double) position_min.values[axis]) ||
                !std::isfinite((double) position_max.values[axis]) ||
                position_max.values[axis] <= position_min.values[axis]) {
                return false;
            }
        }

        for (int level = 0; level < npm_grid_levels; ++level) {
            const std::string name = "npm_grid_" + std::to_string(level);
            auto it = tensors.find(name);
            if (it == tensors.end()) {
                return false;
            }
            const std::vector<size_t> &dims = it->second.dims;
            if (dims.size() != 4 || dims[0] < 2 || dims[0] != dims[1] ||
                dims[0] != dims[2] || dims[3] != (size_t) npm_grid_features) {
                return false;
            }
        }

        const size_t npm_input_dim =
            (size_t) npm_grid_levels * (size_t) npm_grid_features + 19u;
        for (int layer = 0; layer < depth; ++layer) {
            const std::string prefix = "npm_decoder_" + std::to_string(layer);
            const size_t input_dim = layer == 0 ? npm_input_dim : (size_t) hidden_dim;
            if (!validate_tensor(prefix + "_weight", { (size_t) hidden_dim, input_dim }) ||
                !validate_tensor(prefix + "_bias", { (size_t) hidden_dim })) {
                return false;
            }
        }
        return validate_tensor(
                   "npm_output_weight",
                   { (size_t) (vmf_components * 4), (size_t) hidden_dim }
               ) &&
               validate_tensor(
                   "npm_output_bias",
                   { (size_t) (vmf_components * 4) }
               );
    }

    bool has_tensor(const std::string &name) const {
        return tensors.find(name) != tensors.end();
    }

    bool validate_tensor(const std::string &name, std::initializer_list<size_t> expected_dims) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            return false;
        }

        const auto &dims = it->second.dims;
        if (dims.size() != expected_dims.size()) {
            return false;
        }

        return std::equal(expected_dims.begin(), expected_dims.end(), dims.begin());
    }

    const TensorData &tensor(const std::string &name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("missing tensor in masked diffusion runtime: " + name);
        }
        return it->second;
    }

    Prediction prepare_ordered_context(const std::vector<Scalar> &condition) const {
        Prediction context;
        const std::vector<Scalar> zero_direction(3, (Scalar) 0.f);
        const std::vector<int> all_masked_bits(chain_type_bits, 2);
        context = predict_diffusion_state(
            condition,
            zero_direction,
            chain_length_classes,
            all_masked_bits,
            (Scalar) 1.f
        );
        if (!context.valid) {
            return Prediction();
        }

        // This is deliberately not a legacy-ready Prediction. The current MPG
        // integrator mixes only a length PMF; exposing this as valid there would
        // pair a sampled length with independently decoded bits/direction.
        context.valid = false;
        context.ordered_context_valid = true;
        context.ordered_condition = condition;
        return context;
    }

    Prediction predict_diffusion_state(
        const std::vector<Scalar> &condition,
        const std::vector<Scalar> &noisy_direction,
        int masked_chain_length,
        const std::vector<int> &masked_chain_type_bits,
        Scalar timestep
    ) const {
        if (backbone_type == "mlp" && runtime_input_mode == "full_mlp") {
            return predict_mlp_state(
                condition,
                noisy_direction,
                masked_chain_length,
                masked_chain_type_bits,
                timestep
            );
        }
        if (backbone_type == "transformer" && runtime_input_mode == "full_transformer") {
            return predict_transformer_state(
                condition,
                noisy_direction,
                masked_chain_length,
                masked_chain_type_bits,
                timestep
            );
        }
        if (backbone_type == "dit" && runtime_input_mode == "full_dit") {
            return predict_dit_state(
                condition,
                noisy_direction,
                masked_chain_length,
                masked_chain_type_bits,
                timestep
            );
        }
        return Prediction();
    }

    Prediction predict_single_step(const std::vector<Scalar> &condition) const {
        Prediction prediction;
        std::vector<Scalar> chain_length_hidden;
        std::vector<Scalar> direction_hidden;
        std::vector<Scalar> chain_type_logits;

        if (backbone_type == "transformer") {
            std::vector<std::vector<Scalar>> tokens = run_transformer_backbone(condition);
            if ((int) tokens.size() < 3 + chain_type_bits) {
                return prediction;
            }
            direction_hidden    = tokens[1];
            chain_length_hidden = tokens[2];
            chain_type_logits.resize(chain_type_bits);
            for (int bit = 0; bit < chain_type_bits; ++bit) {
                std::vector<Scalar> bit_logit = linear(
                    tokens[3 + bit],
                    tensor("chain_type_token_head_weight"),
                    tensor("chain_type_token_head_bias")
                );
                chain_type_logits[bit] = bit_logit[0];
            }
        } else if (backbone_type == "dit") {
            const std::vector<Scalar> zero_direction(3, (Scalar) 0.f);
            const std::vector<int> all_masked_bits(chain_type_bits, 2);
            return predict_dit_state(
                condition,
                zero_direction,
                chain_length_classes,
                all_masked_bits,
                (Scalar) 0.f
            );
        } else if (backbone_type == "mlp") {
            std::vector<Scalar> hidden = run_mlp_backbone(condition);
            chain_length_hidden = hidden;
            direction_hidden    = hidden;
            chain_type_logits =
                linear(hidden, tensor("chain_type_head_weight"), tensor("chain_type_head_bias"));
        } else if (backbone_type == "plain_mlp") {
            std::vector<Scalar> hidden = run_plain_mlp_backbone(condition);
            chain_length_hidden = hidden;
            direction_hidden    = hidden;
            chain_type_logits =
                linear(hidden, tensor("chain_type_head_weight"), tensor("chain_type_head_bias"));
        } else {
            return prediction;
        }

        return make_prediction(chain_length_hidden, direction_hidden, chain_type_logits);
    }

    Prediction predict_npm_adapted_from_condition(
        const std::vector<Scalar> &condition
    ) const {
        if ((int) condition.size() != condition_dim || condition_dim < 13) {
            return Prediction();
        }
        const Vector3f position(condition[0], condition[1], condition[2]);
        const Vector3f delta(condition[6], condition[7], condition[8]);
        const Vector3f normal(condition[10], condition[11], condition[12]);
        return run_npm_adapted(
            position,
            normalize_vector3_or_zero(delta),
            normal,
            npm_roughness_default
        );
    }

    std::vector<Scalar> npm_real_sh_degree2(const Vector3f &direction) const {
        const Vector3f unit = normalize_vector3_or_zero(direction);
        const Scalar x = unit.x();
        const Scalar y = unit.y();
        const Scalar z = unit.z();
        return {
            (Scalar) 0.28209479177387814,
            (Scalar) 0.4886025119029199 * y,
            (Scalar) 0.4886025119029199 * z,
            (Scalar) 0.4886025119029199 * x,
            (Scalar) 1.0925484305920792 * x * y,
            (Scalar) 1.0925484305920792 * y * z,
            (Scalar) 0.31539156525252005 * ((Scalar) 3.f * z * z - (Scalar) 1.f),
            (Scalar) 1.0925484305920792 * x * z,
            (Scalar) 0.5462742152960396 * (x * x - y * y),
        };
    }

    std::vector<Scalar> run_npm_grid_encoding(const Vector3f &position) const {
        const TensorData &position_min = tensor("npm_position_min");
        const TensorData &position_max = tensor("npm_position_max");
        std::array<Scalar, 3> normalized;
        for (size_t axis = 0; axis < 3; ++axis) {
            const Scalar extent = std::max(
                position_max.values[axis] - position_min.values[axis],
                (Scalar) 1e-8f
            );
            normalized[axis] = std::min(
                std::max(
                    (position[axis] - position_min.values[axis]) / extent,
                    (Scalar) 0.f
                ),
                (Scalar) 1.f
            );
        }

        std::vector<Scalar> encoded;
        encoded.reserve((size_t) npm_grid_levels * (size_t) npm_grid_features);
        for (int level = 0; level < npm_grid_levels; ++level) {
            const TensorData &grid = tensor("npm_grid_" + std::to_string(level));
            const size_t resolution = grid.dims[0];
            std::array<size_t, 3> lower;
            std::array<size_t, 3> upper;
            std::array<Scalar, 3> fraction;
            for (size_t axis = 0; axis < 3; ++axis) {
                const Scalar scaled = normalized[axis] * (Scalar) (resolution - 1);
                lower[axis] = (size_t) std::floor(scaled);
                upper[axis] = std::min(lower[axis] + 1, resolution - 1);
                fraction[axis] = scaled - (Scalar) lower[axis];
            }
            for (int feature = 0; feature < npm_grid_features; ++feature) {
                Scalar value = (Scalar) 0.f;
                for (int x_high = 0; x_high < 2; ++x_high) {
                    for (int y_high = 0; y_high < 2; ++y_high) {
                        for (int z_high = 0; z_high < 2; ++z_high) {
                            const size_t x = x_high ? upper[0] : lower[0];
                            const size_t y = y_high ? upper[1] : lower[1];
                            const size_t z = z_high ? upper[2] : lower[2];
                            const Scalar wx = x_high ? fraction[0] : (Scalar) 1.f - fraction[0];
                            const Scalar wy = y_high ? fraction[1] : (Scalar) 1.f - fraction[1];
                            const Scalar wz = z_high ? fraction[2] : (Scalar) 1.f - fraction[2];
                            const size_t index =
                                (((x * resolution + y) * resolution + z) *
                                 (size_t) npm_grid_features) +
                                (size_t) feature;
                            value += wx * wy * wz * grid.values[index];
                        }
                    }
                }
                encoded.push_back(value);
            }
        }
        return encoded;
    }

    Prediction run_npm_adapted(
        const Vector3f &position,
        const Vector3f &outgoing_direction,
        const Vector3f &normal,
        Scalar roughness
    ) const {
        Prediction prediction;
        std::vector<Scalar> features = run_npm_grid_encoding(position);
        const std::vector<Scalar> direction_sh = npm_real_sh_degree2(outgoing_direction);
        const std::vector<Scalar> normal_sh = npm_real_sh_degree2(normal);
        features.insert(features.end(), direction_sh.begin(), direction_sh.end());
        features.insert(features.end(), normal_sh.begin(), normal_sh.end());
        features.push_back(roughness);

        std::vector<Scalar> hidden = features;
        for (int layer = 0; layer < depth; ++layer) {
            const std::string prefix = "npm_decoder_" + std::to_string(layer);
            hidden = linear(
                hidden,
                tensor(prefix + "_weight"),
                tensor(prefix + "_bias")
            );
            for (Scalar &value : hidden) {
                value = std::max(value, (Scalar) 0.f);
            }
        }

        const std::vector<Scalar> raw = linear(
            hidden,
            tensor("npm_output_weight"),
            tensor("npm_output_bias")
        );
        std::vector<Scalar> logits((size_t) vmf_components, (Scalar) 0.f);
        prediction.vmf_means.reserve((size_t) vmf_components);
        prediction.vmf_kappas.reserve((size_t) vmf_components);
        for (int component = 0; component < vmf_components; ++component) {
            const size_t offset = (size_t) component * 4u;
            logits[(size_t) component] = raw[offset];
            const Scalar raw_kappa = std::min(
                std::max(
                    raw[offset + 1],
                    (Scalar) std::log((double) vmf_min_kappa)
                ),
                (Scalar) std::log((double) vmf_max_kappa)
            );
            prediction.vmf_kappas.push_back(std::exp(raw_kappa));
            const Scalar theta = (Scalar) math::Pi<Scalar> * logistic(raw[offset + 2]);
            const Scalar phi = (Scalar) 2.f * (Scalar) math::Pi<Scalar> * logistic(raw[offset + 3]);
            const Scalar sin_theta = std::sin(theta);
            prediction.vmf_means.emplace_back(
                sin_theta * std::cos(phi),
                sin_theta * std::sin(phi),
                std::cos(theta)
            );
        }
        prediction.vmf_weights = softmax(logits);
        Vector3f expected((Scalar) 0.f);
        for (int component = 0; component < vmf_components; ++component) {
            expected += prediction.vmf_means[(size_t) component] *
                        prediction.vmf_weights[(size_t) component];
        }
        prediction.direction = normalize_vector3_or_zero(expected);
        prediction.direction_only = true;
        prediction.valid = true;
        return prediction;
    }

    Prediction make_prediction(
        const std::vector<Scalar> &chain_length_hidden,
        const std::vector<Scalar> &direction_hidden,
        const std::vector<Scalar> &chain_type_logits
    ) const {
        Prediction prediction;
        prediction.chain_type_logits = chain_type_logits;
        prediction.chain_length_logits =
            linear(chain_length_hidden, tensor("chain_length_head_weight"), tensor("chain_length_head_bias"));
        prediction.chain_length_probs = softmax(prediction.chain_length_logits);
        fill_direction_prediction(prediction, direction_hidden);
        prediction.valid = true;
        return prediction;
    }


    std::vector<Scalar> run_plain_mlp_backbone(const std::vector<Scalar> &condition) const {
        std::vector<Scalar> hidden = condition;
        for (int layer_index = 0; layer_index < depth; ++layer_index) {
            const std::string prefix = "plain_layer_" + std::to_string(layer_index);
            hidden = linear(hidden, tensor(prefix + "_weight"), tensor(prefix + "_bias"));
            for (Scalar &value : hidden) {
                value = value / ((Scalar) 1 + std::exp(-value));
            }
        }
        return hidden;
    }

    Prediction predict_mlp_multistep(const std::vector<Scalar> &condition, int diffusion_steps) const {
        int masked_chain_length = chain_length_classes;
        std::vector<int> masked_chain_type_bits(chain_type_bits, 2);
        std::vector<Scalar> noisy_direction(3, 0.f);

        if (diffusion_steps <= 1) {
            return predict_mlp_state(condition, noisy_direction, masked_chain_length, masked_chain_type_bits, (Scalar) 0.f);
        }

        Prediction step_prediction;
        for (int step = diffusion_steps; step >= 1; --step) {
            const Scalar timestep = (Scalar) step / (Scalar) diffusion_steps;
            step_prediction = predict_mlp_state(
                condition,
                noisy_direction,
                masked_chain_length,
                masked_chain_type_bits,
                timestep
            );
            if (!step_prediction.valid) {
                return step_prediction;
            }
            masked_chain_length = argmax(step_prediction.chain_length_logits);
            masked_chain_type_bits.assign(chain_type_bits, 0);
            for (int bit = 0; bit < chain_type_bits; ++bit) {
                masked_chain_type_bits[bit] = step_prediction.chain_type_logits[bit] >= (Scalar) 0.f ? 1 : 0;
            }
            noisy_direction = vector3_to_std(step_prediction.direction);
        }

        return predict_mlp_state(
            condition,
            noisy_direction,
            masked_chain_length,
            masked_chain_type_bits,
            (Scalar) 0.f
        );
    }

    Prediction predict_mlp_state(
        const std::vector<Scalar> &condition,
        const std::vector<Scalar> &noisy_direction,
        int masked_chain_length,
        const std::vector<int> &masked_chain_type_bits,
        Scalar timestep
    ) const {
        std::vector<Scalar> hidden = run_mlp_backbone_full(
            condition,
            noisy_direction,
            masked_chain_length,
            masked_chain_type_bits,
            timestep
        );
        std::vector<Scalar> chain_type_logits =
            linear(hidden, tensor("chain_type_head_weight"), tensor("chain_type_head_bias"));
        return make_prediction(hidden, hidden, chain_type_logits);
    }

    Prediction predict_transformer_state(
        const std::vector<Scalar> &condition,
        const std::vector<Scalar> &noisy_direction,
        int masked_chain_length,
        const std::vector<int> &masked_chain_type_bits,
        Scalar timestep
    ) const {
        Prediction prediction;
        std::vector<std::vector<Scalar>> tokens = run_transformer_backbone_full(
            condition,
            noisy_direction,
            masked_chain_length,
            masked_chain_type_bits,
            timestep
        );
        if ((int) tokens.size() < 3 + chain_type_bits) {
            return prediction;
        }

        std::vector<Scalar> chain_type_logits(chain_type_bits, (Scalar) 0.f);
        for (int bit = 0; bit < chain_type_bits; ++bit) {
            const std::vector<Scalar> bit_logit = linear(
                tokens[3 + bit],
                tensor("chain_type_token_head_weight"),
                tensor("chain_type_token_head_bias")
            );
            chain_type_logits[bit] = bit_logit[0];
        }
        return make_prediction(tokens[2], tokens[1], chain_type_logits);
    }

    Prediction predict_dit_state(
        const std::vector<Scalar> &condition,
        const std::vector<Scalar> &noisy_direction,
        int masked_chain_length,
        const std::vector<int> &masked_chain_type_bits,
        Scalar timestep
    ) const {
        Prediction prediction;
        std::vector<std::vector<Scalar>> tokens = run_dit_backbone_full(
            condition,
            noisy_direction,
            masked_chain_length,
            masked_chain_type_bits,
            timestep
        );
        if ((int) tokens.size() < 2 + chain_type_bits) {
            return prediction;
        }

        std::vector<Scalar> chain_type_logits(chain_type_bits, (Scalar) 0.f);
        for (int bit = 0; bit < chain_type_bits; ++bit) {
            const std::vector<Scalar> bit_logit = linear(
                tokens[2 + bit],
                tensor("chain_type_token_head_weight"),
                tensor("chain_type_token_head_bias")
            );
            chain_type_logits[bit] = bit_logit[0];
        }
        return make_prediction(tokens[1], tokens[0], chain_type_logits);
    }

    std::vector<Scalar> run_mlp_backbone_full(
        const std::vector<Scalar> &condition,
        const std::vector<Scalar> &noisy_direction,
        int masked_chain_length,
        const std::vector<int> &masked_chain_type_bits,
        Scalar timestep
    ) const {
        std::vector<Scalar> input;
        input.reserve((size_t) (condition_dim + 3 + time_embedding_dim + token_embedding_dim + token_embedding_dim * chain_type_bits));
        input.insert(input.end(), condition.begin(), condition.end());
        for (int i = 0; i < 3; ++i) {
            input.push_back(i < (int) noisy_direction.size() ? noisy_direction[i] : (Scalar) 0.f);
        }

        const std::vector<Scalar> time_embedding = sinusoidal_time_embedding(timestep);
        input.insert(input.end(), time_embedding.begin(), time_embedding.end());

        append_embedding_row(input, tensor("chain_length_embedding_weight"), clamp_index(masked_chain_length, 0, chain_length_classes));
        for (int bit = 0; bit < chain_type_bits; ++bit) {
            const int token = bit < (int) masked_chain_type_bits.size() ? masked_chain_type_bits[bit] : 2;
            append_embedding_row(input, tensor("chain_type_embedding_weight"), clamp_index(token, 0, 2));
        }

        std::vector<Scalar> hidden = linear(input, tensor("input_weight"), tensor("input_bias"));
        apply_silu_inplace(hidden);

        for (int block_index = 0; block_index < depth; ++block_index) {
            const std::vector<Scalar> residual = hidden;
            hidden = layer_norm(
                hidden,
                tensor("block_" + std::to_string(block_index) + "_norm_weight"),
                tensor("block_" + std::to_string(block_index) + "_norm_bias")
            );
            hidden = linear(
                hidden,
                tensor("block_" + std::to_string(block_index) + "_linear1_weight"),
                tensor("block_" + std::to_string(block_index) + "_linear1_bias")
            );
            apply_silu_inplace(hidden);
            hidden = linear(
                hidden,
                tensor("block_" + std::to_string(block_index) + "_linear2_weight"),
                tensor("block_" + std::to_string(block_index) + "_linear2_bias")
            );
            for (size_t i = 0; i < hidden.size(); ++i) {
                hidden[i] += residual[i];
            }
        }

        return layer_norm(hidden, tensor("final_norm_weight"), tensor("final_norm_bias"));
    }

    std::vector<Scalar> sinusoidal_time_embedding(Scalar timestep) const {
        std::vector<Scalar> embedding;
        if (time_embedding_dim <= 0) {
            return embedding;
        }
        if (time_embedding_dim == 1) {
            embedding.push_back(timestep);
            return embedding;
        }
        const int half_dim = time_embedding_dim / 2;
        const Scalar scale = std::log((Scalar) 10000.f) / (Scalar) std::max(half_dim - 1, 1);
        embedding.reserve((size_t) time_embedding_dim);
        for (int i = 0; i < half_dim; ++i) {
            const Scalar exponent = std::exp((Scalar) i * -scale);
            embedding.push_back(std::sin(timestep * exponent));
        }
        for (int i = 0; i < half_dim; ++i) {
            const Scalar exponent = std::exp((Scalar) i * -scale);
            embedding.push_back(std::cos(timestep * exponent));
        }
        if ((int) embedding.size() < time_embedding_dim) {
            embedding.push_back((Scalar) 0.f);
        }
        return embedding;
    }

    void append_embedding_row(std::vector<Scalar> &target, const TensorData &embedding, int row) const {
        const int row_dim = (int) embedding.dims[1];
        const int clamped_row = clamp_index(row, 0, (int) embedding.dims[0] - 1);
        const size_t offset = (size_t) clamped_row * (size_t) row_dim;
        for (int dim = 0; dim < row_dim; ++dim) {
            target.push_back(embedding.values[offset + (size_t) dim]);
        }
    }

    std::vector<Scalar> embedding_row(const TensorData &embedding, int row) const {
        std::vector<Scalar> result;
        result.reserve(embedding.dims.size() >= 2 ? embedding.dims[1] : 0u);
        append_embedding_row(result, embedding, row);
        return result;
    }

    int clamp_index(int value, int min_value, int max_value) const {
        return std::max(min_value, std::min(value, max_value));
    }

    int argmax(const std::vector<Scalar> &values) const {
        if (values.empty()) {
            return 0;
        }
        return (int) std::distance(values.begin(), std::max_element(values.begin(), values.end()));
    }

    Scalar clamp_unit_sample(Scalar value) const {
        return std::min(
            std::max(value, (Scalar) 0.f),
            std::nextafter((Scalar) 1.f, (Scalar) 0.f)
        );
    }

    bool is_unit_sample(Scalar value) const {
        return std::isfinite((double) value) &&
               value >= (Scalar) 0.f && value <= (Scalar) 1.f;
    }

    Scalar sigmoid(Scalar value) const {
        if (value >= (Scalar) 0.f) {
            const Scalar z = std::exp(-value);
            return (Scalar) 1.f / ((Scalar) 1.f + z);
        }
        const Scalar z = std::exp(value);
        return z / ((Scalar) 1.f + z);
    }

    Scalar ordered_bit_probability_one(Scalar logit) const {
        // Mathematical sigmoid probabilities have open support, but finite
        // precision can round an extreme logit to exactly zero or one.  Keep
        // both descriptor strata reachable and use this same probability in
        // sampling and PDF evaluation.
        const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
        return std::min(
            std::max(sigmoid(logit), epsilon),
            (Scalar) 1.f - epsilon
        );
    }

    Scalar safe_log_probability(Scalar probability) const {
        return std::log(std::max(probability, (Scalar) 1e-30f));
    }

    Scalar positive_chain_length_probability(
        const std::vector<Scalar> &probabilities,
        int chain_length
    ) const {
        if (chain_length <= 0 || chain_length >= (int) probabilities.size()) {
            return (Scalar) 0.f;
        }
        Scalar valid_mass = 0.f;
        for (size_t index = 1; index < probabilities.size(); ++index) {
            valid_mass += probabilities[index];
        }
        if (valid_mass <= (Scalar) 1e-30f) {
            return (Scalar) 0.f;
        }
        return probabilities[(size_t) chain_length] / valid_mass;
    }

    int sample_positive_chain_length(
        const std::vector<Scalar> &probabilities,
        Scalar sample,
        Scalar &probability
    ) const {
        probability = 0.f;
        Scalar valid_mass = 0.f;
        for (size_t index = 1; index < probabilities.size(); ++index) {
            valid_mass += probabilities[index];
        }
        if (valid_mass <= (Scalar) 1e-30f) {
            return 1;
        }

        const Scalar target = clamp_unit_sample(sample) * valid_mass;
        Scalar cumulative = 0.f;
        for (size_t index = 1; index < probabilities.size(); ++index) {
            cumulative += probabilities[index];
            if (target <= cumulative || index + 1 == probabilities.size()) {
                probability = probabilities[index] / valid_mass;
                return (int) index;
            }
        }
        return 1;
    }

    std::vector<Scalar> vector3_to_std(const Vector3f &value) const {
        return { value.x(), value.y(), value.z() };
    }

    std::vector<Scalar> run_mlp_backbone(const std::vector<Scalar> &condition) const {
        std::vector<Scalar> hidden = linear(condition, tensor("input_weight"), tensor("input_bias"));
        apply_silu_inplace(hidden);

        for (int block_index = 0; block_index < depth; ++block_index) {
            const std::vector<Scalar> residual = hidden;
            hidden = layer_norm(
                hidden,
                tensor("block_" + std::to_string(block_index) + "_norm_weight"),
                tensor("block_" + std::to_string(block_index) + "_norm_bias")
            );
            hidden = linear(
                hidden,
                tensor("block_" + std::to_string(block_index) + "_linear1_weight"),
                tensor("block_" + std::to_string(block_index) + "_linear1_bias")
            );
            apply_silu_inplace(hidden);
            hidden = linear(
                hidden,
                tensor("block_" + std::to_string(block_index) + "_linear2_weight"),
                tensor("block_" + std::to_string(block_index) + "_linear2_bias")
            );
            for (size_t i = 0; i < hidden.size(); ++i) {
                hidden[i] += residual[i];
            }
        }

        return layer_norm(hidden, tensor("final_norm_weight"), tensor("final_norm_bias"));
    }

    std::vector<std::vector<Scalar>> run_dit_backbone_full(
        const std::vector<Scalar> &condition,
        const std::vector<Scalar> &noisy_direction,
        int masked_chain_length,
        const std::vector<int> &masked_chain_type_bits,
        Scalar timestep
    ) const {
        std::vector<Scalar> conditioning = linear(
            condition,
            tensor("dit_condition_weight"),
            tensor("dit_condition_bias")
        );
        std::vector<Scalar> time_condition = linear(
            sinusoidal_time_embedding(timestep),
            tensor("dit_time_linear1_weight"),
            tensor("dit_time_linear1_bias")
        );
        apply_silu_inplace(time_condition);
        time_condition = linear(
            time_condition,
            tensor("dit_time_linear2_weight"),
            tensor("dit_time_linear2_bias")
        );
        add_inplace(conditioning, time_condition);

        std::vector<std::vector<Scalar>> tokens(
            token_count,
            std::vector<Scalar>(hidden_dim, (Scalar) 0.f)
        );
        std::vector<Scalar> direction(3, (Scalar) 0.f);
        for (int dim = 0; dim < 3; ++dim) {
            if (dim < (int) noisy_direction.size()) {
                direction[dim] = noisy_direction[dim];
            }
        }
        tokens[0] = linear(
            direction,
            tensor("dit_direction_token_weight"),
            tensor("dit_direction_token_bias")
        );
        tokens[1] = linear(
            embedding_row(
                tensor("chain_length_embedding_weight"),
                clamp_index(masked_chain_length, 0, chain_length_classes)
            ),
            tensor("chain_length_token_weight"),
            tensor("chain_length_token_bias")
        );
        for (int bit = 0; bit < chain_type_bits; ++bit) {
            const int token = bit < (int) masked_chain_type_bits.size()
                ? masked_chain_type_bits[bit]
                : 2;
            tokens[2 + bit] = linear(
                embedding_row(
                    tensor("chain_type_embedding_weight"),
                    clamp_index(token, 0, 2)
                ),
                tensor("chain_type_token_weight"),
                tensor("chain_type_token_bias")
            );
        }

        const TensorData &position = tensor("dit_position_embedding");
        for (int token_index = 0; token_index < token_count; ++token_index) {
            for (int dim = 0; dim < hidden_dim; ++dim) {
                tokens[token_index][dim] += position.values[token_index * hidden_dim + dim];
            }
        }

        for (int block_index = 0; block_index < depth; ++block_index) {
            const std::string prefix = "dit_" + std::to_string(block_index);
            std::vector<Scalar> modulation_input = conditioning;
            apply_silu_inplace(modulation_input);
            const std::vector<Scalar> modulation = linear(
                modulation_input,
                tensor(prefix + "_adaln_weight"),
                tensor(prefix + "_adaln_bias")
            );
            const std::vector<Scalar> shift_attention =
                slice_vector(modulation, 0, hidden_dim);
            const std::vector<Scalar> scale_attention =
                slice_vector(modulation, hidden_dim, hidden_dim);
            const std::vector<Scalar> gate_attention =
                slice_vector(modulation, 2 * hidden_dim, hidden_dim);
            const std::vector<Scalar> shift_mlp =
                slice_vector(modulation, 3 * hidden_dim, hidden_dim);
            const std::vector<Scalar> scale_mlp =
                slice_vector(modulation, 4 * hidden_dim, hidden_dim);
            const std::vector<Scalar> gate_mlp =
                slice_vector(modulation, 5 * hidden_dim, hidden_dim);

            std::vector<std::vector<Scalar>> attention_inputs(tokens.size());
            for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
                attention_inputs[token_index] = layer_norm_no_affine(tokens[token_index]);
                modulate_inplace(
                    attention_inputs[token_index],
                    shift_attention,
                    scale_attention
                );
            }
            const std::vector<std::vector<Scalar>> attended =
                multi_head_attention(attention_inputs, prefix);
            for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
                gated_add_inplace(
                    tokens[token_index],
                    attended[token_index],
                    gate_attention
                );
            }

            for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
                std::vector<Scalar> mlp_input = layer_norm_no_affine(tokens[token_index]);
                modulate_inplace(mlp_input, shift_mlp, scale_mlp);
                std::vector<Scalar> update = linear(
                    mlp_input,
                    tensor(prefix + "_linear1_weight"),
                    tensor(prefix + "_linear1_bias")
                );
                apply_gelu_inplace(update);
                update = linear(
                    update,
                    tensor(prefix + "_linear2_weight"),
                    tensor(prefix + "_linear2_bias")
                );
                gated_add_inplace(tokens[token_index], update, gate_mlp);
            }
        }

        std::vector<Scalar> final_modulation_input = conditioning;
        apply_silu_inplace(final_modulation_input);
        const std::vector<Scalar> final_modulation = linear(
            final_modulation_input,
            tensor("dit_final_adaln_weight"),
            tensor("dit_final_adaln_bias")
        );
        const std::vector<Scalar> final_shift =
            slice_vector(final_modulation, 0, hidden_dim);
        const std::vector<Scalar> final_scale =
            slice_vector(final_modulation, hidden_dim, hidden_dim);
        for (std::vector<Scalar> &token : tokens) {
            token = layer_norm_no_affine(token);
            modulate_inplace(token, final_shift, final_scale);
        }
        return tokens;
    }

    std::vector<std::vector<Scalar>> run_transformer_backbone(
        const std::vector<Scalar> &condition
    ) const {
        std::vector<std::vector<Scalar>> tokens(token_count, std::vector<Scalar>(hidden_dim, 0.f));
        tokens[0] = linear(condition, tensor("condition_token_weight"), tensor("condition_token_bias"));
        apply_silu_inplace(tokens[0]);

        const TensorData &constants = tensor("runtime_token_constants");
        for (int token_index = 1; token_index < token_count; ++token_index) {
            for (int dim = 0; dim < hidden_dim; ++dim) {
                tokens[token_index][dim] = constants.values[(token_index - 1) * hidden_dim + dim];
            }
        }

        return finish_transformer_backbone(std::move(tokens));
    }

    std::vector<std::vector<Scalar>> run_transformer_backbone_full(
        const std::vector<Scalar> &condition,
        const std::vector<Scalar> &noisy_direction,
        int masked_chain_length,
        const std::vector<int> &masked_chain_type_bits,
        Scalar timestep
    ) const {
        std::vector<std::vector<Scalar>> tokens(
            token_count,
            std::vector<Scalar>(hidden_dim, (Scalar) 0.f)
        );
        tokens[0] = linear(
            condition,
            tensor("condition_token_weight"),
            tensor("condition_token_bias")
        );
        apply_silu_inplace(tokens[0]);

        std::vector<Scalar> direction_time;
        direction_time.reserve((size_t) (3 + time_embedding_dim));
        for (int dim = 0; dim < 3; ++dim) {
            direction_time.push_back(
                dim < (int) noisy_direction.size()
                    ? noisy_direction[dim]
                    : (Scalar) 0.f
            );
        }
        const std::vector<Scalar> time_embedding = sinusoidal_time_embedding(timestep);
        direction_time.insert(
            direction_time.end(),
            time_embedding.begin(),
            time_embedding.end()
        );
        tokens[1] = linear(
            direction_time,
            tensor("direction_time_token_weight"),
            tensor("direction_time_token_bias")
        );
        apply_silu_inplace(tokens[1]);

        tokens[2] = linear(
            embedding_row(
                tensor("chain_length_embedding_weight"),
                clamp_index(masked_chain_length, 0, chain_length_classes)
            ),
            tensor("chain_length_token_weight"),
            tensor("chain_length_token_bias")
        );
        for (int bit = 0; bit < chain_type_bits; ++bit) {
            const int token = bit < (int) masked_chain_type_bits.size()
                ? masked_chain_type_bits[bit]
                : 2;
            tokens[3 + bit] = linear(
                embedding_row(
                    tensor("chain_type_embedding_weight"),
                    clamp_index(token, 0, 2)
                ),
                tensor("chain_type_token_weight"),
                tensor("chain_type_token_bias")
            );
        }

        return finish_transformer_backbone(std::move(tokens));
    }

    std::vector<std::vector<Scalar>> finish_transformer_backbone(
        std::vector<std::vector<Scalar>> tokens
    ) const {

        const TensorData &position = tensor("transformer_position_embedding");
        for (int token_index = 0; token_index < token_count; ++token_index) {
            for (int dim = 0; dim < hidden_dim; ++dim) {
                tokens[token_index][dim] += position.values[token_index * hidden_dim + dim];
            }
        }

        for (int block_index = 0; block_index < depth; ++block_index) {
            const std::string prefix = "transformer_" + std::to_string(block_index);
            std::vector<std::vector<Scalar>> normed(tokens.size());
            for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
                normed[token_index] = layer_norm(
                    tokens[token_index],
                    tensor(prefix + "_norm1_weight"),
                    tensor(prefix + "_norm1_bias")
                );
            }
            std::vector<std::vector<Scalar>> attended = multi_head_attention(normed, prefix);
            for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
                add_inplace(tokens[token_index], attended[token_index]);
            }

            for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
                std::vector<Scalar> ffn_input = layer_norm(
                    tokens[token_index],
                    tensor(prefix + "_norm2_weight"),
                    tensor(prefix + "_norm2_bias")
                );
                std::vector<Scalar> hidden = linear(
                    ffn_input,
                    tensor(prefix + "_linear1_weight"),
                    tensor(prefix + "_linear1_bias")
                );
                apply_gelu_inplace(hidden);
                std::vector<Scalar> update = linear(
                    hidden,
                    tensor(prefix + "_linear2_weight"),
                    tensor(prefix + "_linear2_bias")
                );
                add_inplace(tokens[token_index], update);
            }
        }

        for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
            tokens[token_index] = layer_norm(
                tokens[token_index],
                tensor("final_norm_weight"),
                tensor("final_norm_bias")
            );
        }
        return tokens;
    }

    std::vector<std::vector<Scalar>> multi_head_attention(
        const std::vector<std::vector<Scalar>> &tokens,
        const std::string &prefix
    ) const {
        const int head_dim = hidden_dim / transformer_heads;
        const Scalar scale = (Scalar) 1.f / std::sqrt((Scalar) head_dim);
        std::vector<std::vector<Scalar>> queries(tokens.size());
        std::vector<std::vector<Scalar>> keys(tokens.size());
        std::vector<std::vector<Scalar>> values(tokens.size());

        for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
            std::vector<Scalar> projected = linear(
                tokens[token_index],
                tensor(prefix + "_self_attn_in_proj_weight"),
                tensor(prefix + "_self_attn_in_proj_bias")
            );
            queries[token_index] = slice_vector(projected, 0, hidden_dim);
            keys[token_index]    = slice_vector(projected, hidden_dim, hidden_dim);
            values[token_index]  = slice_vector(projected, hidden_dim * 2, hidden_dim);
        }

        std::vector<std::vector<Scalar>> outputs(tokens.size(), std::vector<Scalar>(hidden_dim, 0.f));
        for (int head = 0; head < transformer_heads; ++head) {
            const int offset = head * head_dim;
            for (size_t query_index = 0; query_index < tokens.size(); ++query_index) {
                std::vector<Scalar> logits(tokens.size(), 0.f);
                Scalar max_logit = -std::numeric_limits<Scalar>::infinity();
                for (size_t key_index = 0; key_index < tokens.size(); ++key_index) {
                    Scalar score = 0.f;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        score += queries[query_index][offset + dim] * keys[key_index][offset + dim];
                    }
                    logits[key_index] = score * scale;
                    max_logit = std::max(max_logit, logits[key_index]);
                }

                Scalar normalizer = 0.f;
                for (Scalar &logit : logits) {
                    logit = std::exp(logit - max_logit);
                    normalizer += logit;
                }
                normalizer = std::max(normalizer, (Scalar) 1e-6f);

                for (size_t key_index = 0; key_index < tokens.size(); ++key_index) {
                    Scalar weight = logits[key_index] / normalizer;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        outputs[query_index][offset + dim] += weight * values[key_index][offset + dim];
                    }
                }
            }
        }

        for (size_t token_index = 0; token_index < outputs.size(); ++token_index) {
            outputs[token_index] = linear(
                outputs[token_index],
                tensor(prefix + "_self_attn_out_proj_weight"),
                tensor(prefix + "_self_attn_out_proj_bias")
            );
        }
        return outputs;
    }

    std::vector<Scalar> linear(
        const std::vector<Scalar> &input,
        const TensorData &weight,
        const TensorData &bias
    ) const {
        std::vector<Scalar> output(weight.dims[0], 0.f);
        for (size_t row = 0; row < output.size(); ++row) {
            Scalar accum = bias.values[row];
            for (size_t col = 0; col < input.size(); ++col) {
                accum += weight.values[row * input.size() + col] * input[col];
            }
            output[row] = accum;
        }
        return output;
    }

    std::vector<Scalar> layer_norm(
        const std::vector<Scalar> &input,
        const TensorData &weight,
        const TensorData &bias
    ) const {
        Scalar mean = 0.f;
        for (Scalar value : input) {
            mean += value;
        }
        mean /= (Scalar) input.size();

        Scalar variance = 0.f;
        for (Scalar value : input) {
            Scalar delta = value - mean;
            variance += delta * delta;
        }
        variance /= (Scalar) input.size();
        Scalar inv_std = rsqrt(variance + layer_norm_eps);

        std::vector<Scalar> output(input.size(), 0.f);
        for (size_t index = 0; index < input.size(); ++index) {
            output[index] = ((input[index] - mean) * inv_std) * weight.values[index] + bias.values[index];
        }
        return output;
    }

    std::vector<Scalar> layer_norm_no_affine(
        const std::vector<Scalar> &input
    ) const {
        Scalar mean = 0.f;
        for (Scalar value : input) {
            mean += value;
        }
        mean /= (Scalar) input.size();

        Scalar variance = 0.f;
        for (Scalar value : input) {
            const Scalar delta = value - mean;
            variance += delta * delta;
        }
        variance /= (Scalar) input.size();
        const Scalar inv_std = rsqrt(variance + layer_norm_eps);

        std::vector<Scalar> output(input.size(), (Scalar) 0.f);
        for (size_t index = 0; index < input.size(); ++index) {
            output[index] = (input[index] - mean) * inv_std;
        }
        return output;
    }

    void modulate_inplace(
        std::vector<Scalar> &values,
        const std::vector<Scalar> &shift,
        const std::vector<Scalar> &scale
    ) const {
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = values[index] * ((Scalar) 1.f + scale[index]) + shift[index];
        }
    }

    void gated_add_inplace(
        std::vector<Scalar> &target,
        const std::vector<Scalar> &source,
        const std::vector<Scalar> &gate
    ) const {
        for (size_t index = 0; index < target.size(); ++index) {
            target[index] += gate[index] * source[index];
        }
    }

    void apply_silu_inplace(std::vector<Scalar> &values) const {
        for (Scalar &value : values) {
            value = value / (1.f + std::exp(-value));
        }
    }

    void apply_gelu_inplace(std::vector<Scalar> &values) const {
        const Scalar inv_sqrt2 = (Scalar) 0.7071067811865475;
        for (Scalar &value : values) {
            value = (Scalar) 0.5f * value * ((Scalar) 1.f + std::erf(value * inv_sqrt2));
        }
    }

    void add_inplace(std::vector<Scalar> &target, const std::vector<Scalar> &source) const {
        for (size_t index = 0; index < target.size(); ++index) {
            target[index] += source[index];
        }
    }

    std::vector<Scalar> slice_vector(
        const std::vector<Scalar> &values,
        int offset,
        int count
    ) const {
        return std::vector<Scalar>(values.begin() + offset, values.begin() + offset + count);
    }

    std::vector<Scalar> softmax(const std::vector<Scalar> &logits) const {
        Scalar max_logit = logits[0];
        for (Scalar value : logits) {
            max_logit = std::max(max_logit, value);
        }

        std::vector<Scalar> probs(logits.size(), 0.f);
        Scalar normalizer = 0.f;
        for (size_t index = 0; index < logits.size(); ++index) {
            probs[index] = std::exp(logits[index] - max_logit);
            normalizer += probs[index];
        }
        normalizer = std::max(normalizer, (Scalar) 1e-6f);
        for (Scalar &value : probs) {
            value /= normalizer;
        }
        return probs;
    }

    Scalar softplus(Scalar value) const {
        if (value > (Scalar) 20.f) {
            return value;
        }
        if (value < (Scalar) -20.f) {
            return std::exp(value);
        }
        return std::log((Scalar) 1.f + std::exp(value));
    }

    Scalar logistic(Scalar value) const {
        if (value >= (Scalar) 0.f) {
            return (Scalar) 1.f / ((Scalar) 1.f + std::exp(-value));
        }
        const Scalar exp_value = std::exp(value);
        return exp_value / ((Scalar) 1.f + exp_value);
    }

    Scalar log_vmf_normalizer(Scalar kappa) const {
        kappa = std::max(kappa, (Scalar) 1e-6f);
        if (kappa < (Scalar) 1e-5f) {
            return -std::log((Scalar) 4.f * math::Pi<Scalar>);
        }
        Scalar log_sinh = 0.f;
        if (kappa > (Scalar) 20.f) {
            log_sinh = kappa - std::log((Scalar) 2.f) + std::log1p(-std::exp((Scalar) -2.f * kappa));
        } else {
            log_sinh = std::log(std::max(std::sinh(kappa), (Scalar) 1e-20f));
        }
        return std::log(kappa) - std::log((Scalar) 4.f * math::Pi<Scalar>) - log_sinh;
    }

    void fill_direction_prediction(Prediction &prediction, const std::vector<Scalar> &hidden) const {
        if (direction_distribution == "vmf_mixture") {
            prediction.vmf_weights = softmax(
                linear(hidden, tensor("vmf_logits_head_weight"), tensor("vmf_logits_head_bias"))
            );
            std::vector<Scalar> mean_values =
                linear(hidden, tensor("vmf_mean_head_weight"), tensor("vmf_mean_head_bias"));
            std::vector<Scalar> kappa_values =
                linear(hidden, tensor("vmf_kappa_head_weight"), tensor("vmf_kappa_head_bias"));
            prediction.vmf_means.clear();
            prediction.vmf_kappas.clear();
            prediction.vmf_means.reserve(vmf_components);
            prediction.vmf_kappas.reserve(vmf_components);

            Vector3f expected(0.f);
            for (int component = 0; component < vmf_components; ++component) {
                std::vector<Scalar> raw_mean = {
                    mean_values[component * 3 + 0],
                    mean_values[component * 3 + 1],
                    mean_values[component * 3 + 2],
                };
                Vector3f mean = normalize_direction(raw_mean);
                Scalar kappa = std::min(
                    softplus(kappa_values[component]) + vmf_min_kappa,
                    vmf_max_kappa
                );
                prediction.vmf_means.push_back(mean);
                prediction.vmf_kappas.push_back(kappa);
                expected += mean * prediction.vmf_weights[component];
            }
            prediction.direction = normalize_vector3(expected);
            return;
        }

        std::vector<Scalar> direction_values =
            linear(hidden, tensor("direction_head_weight"), tensor("direction_head_bias"));
        prediction.direction = normalize_direction(direction_values);
    }

    Vector3f normalize_direction(const std::vector<Scalar> &values) const {
        Vector3f direction(values[0], values[1], values[2]);
        return normalize_vector3(direction);
    }

    Vector3f normalize_vector3(const Vector3f &direction) const {
        Scalar length2 = dot(direction, direction);
        if (length2 <= (Scalar) 1e-12f) {
            return Vector3f(0.f, 0.f, 1.f);
        }
        return direction * rsqrt(length2);
    }

    Vector3f normalize_vector3_or_zero(const Vector3f &direction) const {
        Scalar length2 = dot(direction, direction);
        if (length2 <= (Scalar) 1e-12f) {
            return Vector3f((Scalar) 0.f);
        }
        return direction * rsqrt(length2);
    }

    Vector3f sample_vmf(
        const Vector3f &mean,
        Scalar kappa,
        Scalar u1,
        Scalar u2
    ) const {
        u1 = std::min(std::max(u1, (Scalar) 1e-6f), (Scalar) 1.f - (Scalar) 1e-6f);
        u2 = std::min(std::max(u2, (Scalar) 0.f), (Scalar) 1.f);
        if (kappa < (Scalar) 1e-5f) {
            Scalar z = (Scalar) 1.f - (Scalar) 2.f * u1;
            Scalar r = std::sqrt(std::max((Scalar) 0.f, (Scalar) 1.f - z * z));
            Scalar phi = (Scalar) 2.f * math::Pi<Scalar> * u2;
            return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
        }

        Scalar exp_term = std::exp((Scalar) -2.f * kappa);
        Scalar w = (Scalar) 1.f + std::log(u1 + ((Scalar) 1.f - u1) * exp_term) / kappa;
        w = std::min(std::max(w, (Scalar) -1.f), (Scalar) 1.f);
        Scalar r = std::sqrt(std::max((Scalar) 0.f, (Scalar) 1.f - w * w));
        Scalar phi = (Scalar) 2.f * math::Pi<Scalar> * u2;

        Vector3f helper = std::abs(mean.z()) < (Scalar) 0.999f
                              ? Vector3f(0.f, 0.f, 1.f)
                              : Vector3f(0.f, 1.f, 0.f);
        Vector3f tangent = normalize_vector3(cross(helper, mean));
        Vector3f bitangent = cross(mean, tangent);
        return normalize_vector3(
            tangent * (r * std::cos(phi)) +
            bitangent * (r * std::sin(phi)) +
            mean * w
        );
    }

    Scalar valid_chain_length_mass(const Prediction &prediction) const {
        Scalar total = 0.f;
        for (size_t index = 1; index < prediction.chain_length_probs.size(); ++index) {
            total += prediction.chain_length_probs[index];
        }
        return total;
    }

    bool loaded = false;
    std::string backbone_type = "mlp";
    std::string direction_distribution = "vector";
    int hidden_dim = 0;
    int depth = 0;
    int condition_dim = 0;
    int vmf_components = 0;
    Scalar vmf_min_kappa = 1e-4f;
    Scalar vmf_max_kappa = 128.f;
    int transformer_heads = 0;
    int transformer_feedforward_dim = 0;
    int token_count = 0;
    int time_embedding_dim = 0;
    int token_embedding_dim = 0;
    std::string runtime_input_mode = "condition_only";
    std::string discrete_diffusion_mode = "legacy";
    int chain_length_classes = 0;
    int chain_type_bits = 0;
    Scalar layer_norm_eps = (Scalar) 1e-5f;
    int npm_grid_levels = 0;
    int npm_grid_features = 0;
    int npm_sh_degree = 0;
    Scalar npm_roughness_default = (Scalar) 0.5f;
    std::string npm_direction_source = "condition_delta_proxy";
    std::string npm_descriptor_policy_id;
    std::unordered_map<std::string, TensorData> tensors;
};

NAMESPACE_END(mitsuba)
