#pragma once
#ifdef HAVE_PICOTREE
#include <pico_tree/kd_tree.hpp>

namespace pico_tree_profiler {

// -----------------------------------------------------------------------------
// WARNING: FRAGILE CODE
// This struct replicates the internal memory layout of pico_tree::kd_tree.
// It relies on "Type Punning" via reinterpret_cast.
//
// IF THE LIBRARY UPDATES AND CHANGES MEMBER ORDER, THIS WILL BREAK/CRASH.
// WORKS WITH PICOTREE VERSION v1.0.0, AND CURRENT CMAKE BUILD WILL FETCH THIS VERSION.
// -----------------------------------------------------------------------------
template <typename TreeType>
struct pico_tree_shadow {
    // Type definitions used inside pico_tree
    using Space    = typename TreeType::space_type;
    using Metric   = typename TreeType::metric_type;
    using Index    = typename TreeType::index_type;
    using SpaceWrapper = pico_tree::internal::space_wrapper<
        pico_tree::internal::remove_reference_wrapper_t<Space>>;
    using NodeType = typename pico_tree::internal::kd_tree_space_tag_traits<
        typename Metric::space_category>::template node_type<Index, typename SpaceWrapper::scalar_type>;
    using DataType = pico_tree::internal::kd_tree_data<NodeType, TreeType::dim>;

    // We match kd_tree memory map exactly, allowing for access to fields via reinterpret_cast
    Space    space;
    Metric   metric;
    DataType data;
};

// Recursive helper to traverse the raw node pointers
template <typename NodePtr>
size_t count_nodes(NodePtr node) {
    if (!node) return 0;
    return node->is_leaf() ? 1 : 1 + count_nodes(node->left) + count_nodes(node->right);
}

// Public memory profiling function
template <typename TreeType>
size_t get_memory_usage(const TreeType& tree) {
    using NodeType = typename pico_tree_shadow<TreeType>::NodeType;

    // Cast opaque tree to our shadow structure
    const auto* shadow = reinterpret_cast<const pico_tree_shadow<TreeType>*>(&tree);

    size_t tree_bytes = sizeof(tree);
    size_t indices_bytes = shadow->data.indices.capacity() * sizeof(typename TreeType::index_type);
    size_t nodes_bytes = count_nodes(shadow->data.root_node) * sizeof(NodeType);

    return tree_bytes + indices_bytes + nodes_bytes;
}

} // namespace pico_tree_profiler

#endif