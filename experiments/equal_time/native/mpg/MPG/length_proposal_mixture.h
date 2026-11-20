#pragma once

#include <cmath>

namespace mitsuba {

inline bool is_valid_bounce_alpha(float bounce_alpha) {
    return std::isfinite((double) bounce_alpha) &&
           bounce_alpha >= 0.f && bounce_alpha <= 1.f;
}

template <typename Float>
inline Float length_proposal_mixture_pdf(Float geometric_pdf,
                                         Float learned_pdf,
                                         float bounce_alpha) {
    return geometric_pdf * bounce_alpha + learned_pdf * (1.f - bounce_alpha);
}

template <typename Scalar>
inline bool should_flip_learned_direction(bool enabled, Scalar receiver_dot) {
    return enabled && receiver_dot < (Scalar) 0.f;
}

} // namespace mitsuba
