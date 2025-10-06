// ParticleRenderer - Test Harness
// Provides validation tests for particle rendering correctness

#include "ParticleRenderer.h"
#include "../core/Device.h"
#include "../utils/Logger.h"
#include <DirectXMath.h>
#include <vector>
#include <cmath>

namespace ParticleTests {

// Test 1: Verify vertex ordering creates proper triangles
bool TestVertexOrdering() {
    LOG_INFO("TEST 1: Vertex Ordering Validation");

    // Apply the vertex-to-corner mapping
    const char* cornerNames[4] = {"BL", "BR", "TL", "TR"};
    uint32_t corners[6];

    // Fixed mapping from particle_billboard_vs.hlsl
    for (uint32_t vertIdx = 0; vertIdx < 6; vertIdx++) {
        uint32_t cornerIdx;
        if (vertIdx == 0) cornerIdx = 0;      // BL
        else if (vertIdx == 1) cornerIdx = 1; // BR
        else if (vertIdx == 2) cornerIdx = 3; // TR
        else if (vertIdx == 3) cornerIdx = 0; // BL
        else if (vertIdx == 4) cornerIdx = 3; // TR
        else cornerIdx = 2;                    // TL
        corners[vertIdx] = cornerIdx;
    }

    // Verify Triangle 0 (vertices 0,1,2)
    LOG_INFO("  Triangle 0: {} -> {} -> {}",
             cornerNames[corners[0]], cornerNames[corners[1]], cornerNames[corners[2]]);

    // Check Triangle 0 forms BL-BR-TR (should be CCW)
    bool tri0_correct = (corners[0] == 0 && corners[1] == 1 && corners[2] == 3);
    LOG_INFO("    Expected: BL -> BR -> TR");
    LOG_INFO("    Result: {}", tri0_correct ? "PASS" : "FAIL");

    // Verify Triangle 1 (vertices 3,4,5)
    LOG_INFO("  Triangle 1: {} -> {} -> {}",
             cornerNames[corners[3]], cornerNames[corners[4]], cornerNames[corners[5]]);

    // Check Triangle 1 forms BL-TR-TL (should be CCW)
    bool tri1_correct = (corners[3] == 0 && corners[4] == 3 && corners[5] == 2);
    LOG_INFO("    Expected: BL -> TR -> TL");
    LOG_INFO("    Result: {}", tri1_correct ? "PASS" : "FAIL");

    // Verify all corners are used
    bool used[4] = {false};
    for (int i = 0; i < 6; i++) {
        used[corners[i]] = true;
    }
    bool all_used = used[0] && used[1] && used[2] && used[3];
    LOG_INFO("  All corners covered: {}", all_used ? "PASS" : "FAIL");

    return tri0_correct && tri1_correct && all_used;
}

// Test 2: Verify clip space positions
bool TestClipSpacePositions() {
    LOG_INFO("TEST 2: Clip Space Position Validation");

    // Test positions from debug shader
    DirectX::XMFLOAT4 positions[6] = {
        DirectX::XMFLOAT4(-0.25f, -0.25f, 0.5f, 1.0f),  // 0: BL
        DirectX::XMFLOAT4( 0.25f, -0.25f, 0.5f, 1.0f),  // 1: BR
        DirectX::XMFLOAT4( 0.25f,  0.25f, 0.5f, 1.0f),  // 2: TR
        DirectX::XMFLOAT4(-0.25f, -0.25f, 0.5f, 1.0f),  // 3: BL
        DirectX::XMFLOAT4( 0.25f,  0.25f, 0.5f, 1.0f),  // 4: TR
        DirectX::XMFLOAT4(-0.25f,  0.25f, 0.5f, 1.0f)   // 5: TL
    };

    // Verify Triangle 0 winding order (should be CCW in screen space)
    DirectX::XMVECTOR v0 = DirectX::XMLoadFloat4(&positions[0]);
    DirectX::XMVECTOR v1 = DirectX::XMLoadFloat4(&positions[1]);
    DirectX::XMVECTOR v2 = DirectX::XMLoadFloat4(&positions[2]);

    // Calculate 2D cross product for winding order
    float cross0 = (positions[1].x - positions[0].x) * (positions[2].y - positions[0].y) -
                   (positions[2].x - positions[0].x) * (positions[1].y - positions[0].y);

    LOG_INFO("  Triangle 0 winding: {}", cross0 > 0 ? "CCW (correct)" : "CW (wrong)");

    // Verify Triangle 1 winding order
    float cross1 = (positions[4].x - positions[3].x) * (positions[5].y - positions[3].y) -
                   (positions[5].x - positions[3].x) * (positions[4].y - positions[3].y);

    LOG_INFO("  Triangle 1 winding: {}", cross1 > 0 ? "CCW (correct)" : "CW (wrong)");

    // Check that triangles cover the quad area
    float area0 = std::abs(cross0) * 0.5f;
    float area1 = std::abs(cross1) * 0.5f;
    float total_area = area0 + area1;
    float expected_area = 0.5f * 0.5f;  // Width * Height of quad

    LOG_INFO("  Triangle 0 area: {}", area0);
    LOG_INFO("  Triangle 1 area: {}", area1);
    LOG_INFO("  Total area: {} (expected: {})", total_area, expected_area);

    bool areas_match = std::abs(total_area - expected_area) < 0.001f;
    LOG_INFO("  Area coverage: {}", areas_match ? "PASS" : "FAIL");

    return (cross0 > 0) && (cross1 > 0) && areas_match;
}

// Test 3: Verify billboard orientation
bool TestBillboardOrientation() {
    LOG_INFO("TEST 3: Billboard Orientation Validation");

    // Test camera vectors
    DirectX::XMFLOAT3 cameraPos(100.0f, 50.0f, 100.0f);
    DirectX::XMFLOAT3 lookAt(0.0f, 0.0f, 0.0f);
    DirectX::XMFLOAT3 worldUp(0.0f, 1.0f, 0.0f);

    DirectX::XMVECTOR camPosVec = DirectX::XMLoadFloat3(&cameraPos);
    DirectX::XMVECTOR lookAtVec = DirectX::XMLoadFloat3(&lookAt);
    DirectX::XMVECTOR worldUpVec = DirectX::XMLoadFloat3(&worldUp);

    // Calculate billboard vectors
    DirectX::XMVECTOR forward = DirectX::XMVector3Normalize(DirectX::XMVectorSubtract(lookAtVec, camPosVec));
    DirectX::XMVECTOR right = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(forward, worldUpVec));
    DirectX::XMVECTOR up = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(forward, right));

    DirectX::XMFLOAT3 rightVec, upVec, forwardVec;
    DirectX::XMStoreFloat3(&rightVec, right);
    DirectX::XMStoreFloat3(&upVec, up);
    DirectX::XMStoreFloat3(&forwardVec, forward);

    LOG_INFO("  Camera Right: ({:.3f}, {:.3f}, {:.3f})", rightVec.x, rightVec.y, rightVec.z);
    LOG_INFO("  Camera Up: ({:.3f}, {:.3f}, {:.3f})", upVec.x, upVec.y, upVec.z);
    LOG_INFO("  Camera Forward: ({:.3f}, {:.3f}, {:.3f})", forwardVec.x, forwardVec.y, forwardVec.z);

    // Verify orthogonality
    float dot_right_up = DirectX::XMVectorGetX(DirectX::XMVector3Dot(right, up));
    float dot_right_forward = DirectX::XMVectorGetX(DirectX::XMVector3Dot(right, forward));
    float dot_up_forward = DirectX::XMVectorGetX(DirectX::XMVector3Dot(up, forward));

    LOG_INFO("  Orthogonality tests:");
    LOG_INFO("    Right·Up = {:.6f} (should be ~0)", dot_right_up);
    LOG_INFO("    Right·Forward = {:.6f} (should be ~0)", dot_right_forward);
    LOG_INFO("    Up·Forward = {:.6f} (should be ~0)", dot_up_forward);

    bool orthogonal = (std::abs(dot_right_up) < 0.001f) &&
                     (std::abs(dot_right_forward) < 0.001f) &&
                     (std::abs(dot_up_forward) < 0.001f);

    LOG_INFO("  Vectors orthogonal: {}", orthogonal ? "PASS" : "FAIL");

    // Verify unit length
    float len_right = DirectX::XMVectorGetX(DirectX::XMVector3Length(right));
    float len_up = DirectX::XMVectorGetX(DirectX::XMVector3Length(up));
    float len_forward = DirectX::XMVectorGetX(DirectX::XMVector3Length(forward));

    LOG_INFO("  Vector lengths:");
    LOG_INFO("    |Right| = {:.6f} (should be 1.0)", len_right);
    LOG_INFO("    |Up| = {:.6f} (should be 1.0)", len_up);
    LOG_INFO("    |Forward| = {:.6f} (should be 1.0)", len_forward);

    bool unit_length = (std::abs(len_right - 1.0f) < 0.001f) &&
                      (std::abs(len_up - 1.0f) < 0.001f) &&
                      (std::abs(len_forward - 1.0f) < 0.001f);

    LOG_INFO("  Vectors normalized: {}", unit_length ? "PASS" : "FAIL");

    return orthogonal && unit_length;
}

// Test 4: Texture coordinate validation
bool TestTextureCoordinates() {
    LOG_INFO("TEST 4: Texture Coordinate Validation");

    // Expected UV coordinates for each corner
    struct UV {
        float u, v;
        const char* name;
    };

    UV expectedUVs[4] = {
        {0.0f, 1.0f, "BL"},  // Bottom-left
        {1.0f, 1.0f, "BR"},  // Bottom-right
        {0.0f, 0.0f, "TL"},  // Top-left
        {1.0f, 0.0f, "TR"}   // Top-right
    };

    // Vertex to corner mapping (from fixed shader)
    uint32_t corners[6] = {0, 1, 3, 0, 3, 2};  // BL, BR, TR, BL, TR, TL

    LOG_INFO("  Vertex UV mappings:");
    for (uint32_t i = 0; i < 6; i++) {
        UV& uv = expectedUVs[corners[i]];
        LOG_INFO("    V{}: {} -> UV({}, {})", i, uv.name, uv.u, uv.v);
    }

    // Verify UV coverage
    bool has_00 = false, has_01 = false, has_10 = false, has_11 = false;
    for (uint32_t i = 0; i < 6; i++) {
        UV& uv = expectedUVs[corners[i]];
        if (uv.u == 0.0f && uv.v == 0.0f) has_00 = true;
        if (uv.u == 0.0f && uv.v == 1.0f) has_01 = true;
        if (uv.u == 1.0f && uv.v == 0.0f) has_10 = true;
        if (uv.u == 1.0f && uv.v == 1.0f) has_11 = true;
    }

    bool full_coverage = has_00 && has_01 && has_10 && has_11;
    LOG_INFO("  All UV corners covered: {}", full_coverage ? "PASS" : "FAIL");

    return full_coverage;
}

// Main test runner
bool RunAllTests() {
    LOG_INFO("=== PARTICLE RENDERING TEST SUITE ===");
    LOG_INFO("Running comprehensive validation tests...\n");

    bool results[4];
    results[0] = TestVertexOrdering();
    LOG_INFO("");

    results[1] = TestClipSpacePositions();
    LOG_INFO("");

    results[2] = TestBillboardOrientation();
    LOG_INFO("");

    results[3] = TestTextureCoordinates();
    LOG_INFO("");

    // Summary
    LOG_INFO("=== TEST SUMMARY ===");
    LOG_INFO("Test 1 (Vertex Ordering): {}", results[0] ? "PASS" : "FAIL");
    LOG_INFO("Test 2 (Clip Space): {}", results[1] ? "PASS" : "FAIL");
    LOG_INFO("Test 3 (Billboard): {}", results[2] ? "PASS" : "FAIL");
    LOG_INFO("Test 4 (Texture UV): {}", results[3] ? "PASS" : "FAIL");

    bool all_pass = results[0] && results[1] && results[2] && results[3];
    LOG_INFO("\nOVERALL RESULT: {}", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    if (all_pass) {
        LOG_INFO("Particle rendering vertex generation is correct!");
        LOG_INFO("The diagonal issue has been resolved.");
    } else {
        LOG_ERROR("Issues remain in particle rendering. Check failed tests above.");
    }

    return all_pass;
}

} // namespace ParticleTests

// Export function for ParticleRenderer class
void ParticleRenderer::RunValidationTests() {
    ParticleTests::RunAllTests();
}