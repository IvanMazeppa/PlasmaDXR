#pragma once

#include <d3d12.h>
#include <vector>
#include "d3dx12/d3dx12.h"

/**
 * BarrierBatcher - Collects D3D12 resource barriers and submits them in batches
 *
 * Performance optimization: Reduces GPU command processor overhead by batching
 * multiple barrier calls into single ResourceBarrier() submissions.
 *
 * Typical savings: 0.1-0.3ms per frame at high barrier counts.
 *
 * Usage:
 *   BarrierBatcher batcher;
 *   batcher.Transition(resource1, STATE_A, STATE_B);
 *   batcher.Transition(resource2, STATE_C, STATE_D);
 *   batcher.UAV(resource3);
 *   batcher.Flush(cmdList);  // Submits all barriers in one call
 */
class BarrierBatcher {
public:
    static constexpr size_t DEFAULT_RESERVE = 16;

    BarrierBatcher() {
        m_barriers.reserve(DEFAULT_RESERVE);
    }

    // Add a state transition barrier
    void Transition(ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES stateBefore,
                    D3D12_RESOURCE_STATES stateAfter,
                    UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
        if (!resource) return;

        // Skip no-op transitions
        if (stateBefore == stateAfter) return;

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = resource;
        barrier.Transition.StateBefore = stateBefore;
        barrier.Transition.StateAfter = stateAfter;
        barrier.Transition.Subresource = subresource;
        m_barriers.push_back(barrier);
    }

    // Add a UAV barrier (synchronize UAV access)
    void UAV(ID3D12Resource* resource) {
        if (!resource) return;

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.UAV.pResource = resource;
        m_barriers.push_back(barrier);
    }

    // Add an aliasing barrier
    void Aliasing(ID3D12Resource* resourceBefore, ID3D12Resource* resourceAfter) {
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Aliasing.pResourceBefore = resourceBefore;
        barrier.Aliasing.pResourceAfter = resourceAfter;
        m_barriers.push_back(barrier);
    }

    // Submit all collected barriers to the command list
    void Flush(ID3D12GraphicsCommandList* cmdList) {
        if (m_barriers.empty() || !cmdList) return;

        cmdList->ResourceBarrier(static_cast<UINT>(m_barriers.size()), m_barriers.data());
        m_barriers.clear();
    }

    // Check if there are pending barriers
    bool HasPending() const { return !m_barriers.empty(); }

    // Get count of pending barriers
    size_t PendingCount() const { return m_barriers.size(); }

    // Clear without submitting (use with caution)
    void Clear() { m_barriers.clear(); }

    // Reserve capacity for expected barrier count
    void Reserve(size_t count) { m_barriers.reserve(count); }

private:
    std::vector<D3D12_RESOURCE_BARRIER> m_barriers;
};
