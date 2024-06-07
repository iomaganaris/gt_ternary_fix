#include "nabla4_gtfn.hpp"

#include <gridtools/common/defs.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/fn/cartesian.hpp>
#include <gridtools/fn/unstructured.hpp>
#include <gridtools/fn/backend/gpu.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/sid/unknown_kind.hpp>
#include <gridtools/storage/adapter/nanobind_adapter.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

using index_type = std::int32_t;

decltype(auto) calculate_nabla4_wrapper_gpu(int repetitions,
    int dry_runs,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> u_vert,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> v_vert,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any>>, std::tuple<ptrdiff_t>> primal_normal_vert_v1,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any>>, std::tuple<ptrdiff_t>> primal_normal_vert_v2,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> z_nabla2_e,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any>>, std::tuple<ptrdiff_t>> inv_vert_vert_length,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any>>, std::tuple<ptrdiff_t>> inv_primal_edge_length,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> z_nabla4_e2,
    std::int32_t horizontal_start,
    std::int32_t horizontal_end,
    std::int32_t vertical_start,
    std::int32_t vertical_end,
    std::pair<nanobind::ndarray<index_type, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> gt_conn_e2c2v,
    std::pair<nanobind::ndarray<index_type, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> gt_conn_e2ecv) {
    return calculate_nabla4(repetitions,
        dry_runs,
        gridtools::sid::rename_numbered_dimensions<generated::Vertex_t, generated::K_t>(
            gridtools::sid::shift_sid_origin(
                gridtools::nanobind::as_sid(u_vert.first, gridtools::nanobind::stride_spec<1, nanobind::any>{}),
                u_vert.second)),
        gridtools::sid::rename_numbered_dimensions<generated::Vertex_t, generated::K_t>(
            gridtools::sid::shift_sid_origin(
                gridtools::nanobind::as_sid(v_vert.first, gridtools::nanobind::stride_spec<1, nanobind::any>{}),
                v_vert.second)),
        gridtools::sid::rename_numbered_dimensions<generated::ECV_t>(gridtools::sid::shift_sid_origin(
            gridtools::nanobind::as_sid(primal_normal_vert_v1.first, gridtools::nanobind::stride_spec<1>{}),
            primal_normal_vert_v1.second)),
        gridtools::sid::rename_numbered_dimensions<generated::ECV_t>(gridtools::sid::shift_sid_origin(
            gridtools::nanobind::as_sid(primal_normal_vert_v2.first, gridtools::nanobind::stride_spec<1>{}),
            primal_normal_vert_v2.second)),
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t, generated::K_t>(gridtools::sid::shift_sid_origin(
            gridtools::nanobind::as_sid(z_nabla2_e.first, gridtools::nanobind::stride_spec<1, nanobind::any>{}),
            z_nabla2_e.second)),
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t>(gridtools::sid::shift_sid_origin(
            gridtools::nanobind::as_sid(inv_vert_vert_length.first, gridtools::nanobind::stride_spec<1>{}),
            inv_vert_vert_length.second)),
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t>(gridtools::sid::shift_sid_origin(
            gridtools::nanobind::as_sid(inv_primal_edge_length.first, gridtools::nanobind::stride_spec<1>{}),
            inv_primal_edge_length.second)),
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t, generated::K_t>(gridtools::sid::shift_sid_origin(
            gridtools::nanobind::as_sid(z_nabla4_e2.first, gridtools::nanobind::stride_spec<1, nanobind::any>{}),
            z_nabla4_e2.second)),
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t, generated::E2C2V_t>(
            gridtools::sid::shift_sid_origin(
                gridtools::nanobind::as_sid(gt_conn_e2c2v.first, gridtools::nanobind::stride_spec<1, nanobind::any>{}),
                gt_conn_e2c2v.second)),
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t, generated::E2ECV_t>(
            gridtools::sid::shift_sid_origin(
                gridtools::nanobind::as_sid(gt_conn_e2ecv.first, gridtools::nanobind::stride_spec<1, nanobind::any>{}),
                gt_conn_e2ecv.second)),
        gridtools::fn::backend::gpu<generated::block_sizes_t>{});
}

NB_MODULE(nabla4_gtfn, module) {
    module.def("calculate_nabla4_gpu", &calculate_nabla4_wrapper_gpu);
}