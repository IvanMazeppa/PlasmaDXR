#include "PINNPhysicsSystem.h"
#include "../utils/Logger.h"
#include <cmath>
#include <chrono>
#include <sstream>
#include <algorithm>  // std::clamp

using namespace DirectX;

PINNPhysicsSystem::PINNPhysicsSystem()
#ifdef ENABLE_ML_FEATURES
    : m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
    LOG_INFO("[PINN] Initializing Physics-Informed Neural Network system...");
}

PINNPhysicsSystem::~PINNPhysicsSystem() {
#ifdef ENABLE_ML_FEATURES
    m_ortSession.reset();
    m_sessionOptions.reset();
    m_ortEnv.reset();
#endif
    LOG_INFO("[PINN] Shutting down PINN physics system");
}

bool PINNPhysicsSystem::Initialize(const std::string& modelPath) {
#ifndef ENABLE_ML_FEATURES
    LOG_WARN("[PINN] ONNX Runtime not available. PINN features disabled.");
    LOG_WARN("[PINN] To enable: Install ONNX Runtime to external/onnxruntime/ and rebuild");
    return false;
#else
    try {
        LOG_INFO("[PINN] Loading trained PINN model from: {}", modelPath);

        // Create ONNX Runtime environment
        m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PlasmaDX-PINN");

        // Configure session options
        m_sessionOptions = std::make_unique<Ort::SessionOptions>();

        // Optimize for high-core-count CPUs (Ryzen 9 5950X has 32 threads)
        // IntraOp: parallelism within a single operator (matrix multiply, etc.)
        // InterOp: parallelism between independent operators
        m_sessionOptions->SetIntraOpNumThreads(16);  // Use 16 threads for tensor ops
        m_sessionOptions->SetInterOpNumThreads(4);   // Use 4 threads for op parallelism
        m_sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        LOG_INFO("[PINN] ONNX Runtime configured: 16 intra-op threads, 4 inter-op threads");

        // Convert path to wide string (Windows requirement)
        std::wstring wideModelPath(modelPath.begin(), modelPath.end());

        // Create inference session
        m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, wideModelPath.c_str(), *m_sessionOptions);

        // Get input/output metadata
        Ort::AllocatorWithDefaultOptions allocator;

        // Input info - detect v1 (1 input, 7D), v2 (2 inputs, 7D+3D), or v3 (1 input, 10D)
        size_t numInputs = m_ortSession->GetInputCount();
        if (numInputs == 1) {
            // Could be v1 (7D spherical) or v3 (10D Cartesian) - determine from input size
            m_isV2Model = false;
            LOG_INFO("[PINN] Detected single-input model (v1 or v3)");
        } else if (numInputs == 2) {
            m_isV2Model = true;
            LOG_INFO("[PINN] Detected v2 model (parameter-conditioned)");
        } else {
            LOG_ERROR("[PINN] Expected 1 or 2 inputs, found {}", numInputs);
            return false;
        }

        // Get first input info (particle_state)
        auto inputName = m_ortSession->GetInputNameAllocated(0, allocator);
        m_inputNames.push_back(std::string(inputName.get()));

        auto inputTypeInfo = m_ortSession->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        m_inputShape = inputTensorInfo.GetShape();

        LOG_INFO("[PINN] Input 0: '{}', shape: [{}, {}]", m_inputNames[0], m_inputShape[0], m_inputShape[1]);

        // If v2 model, get second input info (physics_params)
        if (m_isV2Model) {
            auto paramsInputName = m_ortSession->GetInputNameAllocated(1, allocator);
            m_inputNames.push_back(std::string(paramsInputName.get()));

            auto paramsTypeInfo = m_ortSession->GetInputTypeInfo(1);
            auto paramsTensorInfo = paramsTypeInfo.GetTensorTypeAndShapeInfo();
            m_paramsInputShape = paramsTensorInfo.GetShape();

            LOG_INFO("[PINN] Input 1: '{}', shape: [{}, {}]", m_inputNames[1], m_paramsInputShape[0], m_paramsInputShape[1]);

            if (m_paramsInputShape[1] != 3) {
                LOG_ERROR("[PINN] Invalid params input shape. Expected 3 (M_bh, α, H/R), got {}", m_paramsInputShape[1]);
                return false;
            }
        }

        // Output info
        size_t numOutputs = m_ortSession->GetOutputCount();
        if (numOutputs != 1) {
            LOG_ERROR("[PINN] Expected 1 output, found {}", numOutputs);
            return false;
        }

        auto outputName = m_ortSession->GetOutputNameAllocated(0, allocator);
        m_outputNames.push_back(std::string(outputName.get()));

        auto outputTypeInfo = m_ortSession->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        m_outputShape = outputTensorInfo.GetShape();

        LOG_INFO("[PINN] Output: '{}', shape: [{}, {}]", m_outputNames[0], m_outputShape[0], m_outputShape[1]);

        // Validate model dimensions and detect v3
        if (m_inputShape[1] == 10 && !m_isV2Model) {
            // v3 model: 10D input (x, y, z, vx, vy, vz, t, M_bh, alpha, H_R)
            m_isV3Model = true;
            LOG_INFO("[PINN] Detected v3 model (10D Cartesian input with total forces)");
        } else if (m_inputShape[1] == 7) {
            // v1/v2 model: 7D input (r, θ, φ, v_r, v_θ, v_φ, t)
            m_isV3Model = false;
            LOG_INFO("[PINN] Detected v{} model (7D spherical input)", m_isV2Model ? 2 : 1);
        } else {
            LOG_ERROR("[PINN] Invalid input shape. Expected 7 (v1/v2) or 10 (v3) features, got {}", m_inputShape[1]);
            return false;
        }

        if (m_outputShape[1] != 3) {
            LOG_ERROR("[PINN] Invalid output shape. Expected 3 forces, got {}", m_outputShape[1]);
            return false;
        }

        m_modelLoaded = true;
        if (m_isV3Model) {
            LOG_INFO("[PINN] Successfully loaded PINN v3 model (total forces output)!");
        } else {
            LOG_INFO("[PINN] Successfully loaded PINN model (version {})!", m_isV2Model ? 2 : 1);
        }
        LOG_INFO("[PINN] Hybrid mode: {} (threshold: {:.1f}× R_ISCO)", m_hybridMode ? "ON" : "OFF", m_hybridThresholdRadius / R_ISCO);
        if (m_isV2Model) {
            LOG_INFO("[PINN] Physics params: M_bh={:.2f}, α={:.3f}, H/R={:.3f}",
                m_physicsParams.blackHoleMassNormalized, m_physicsParams.alphaViscosity, m_physicsParams.diskThickness);
        }

        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("[PINN] ONNX Runtime exception: {}", e.what());
        return false;
    }
#endif
}

bool PINNPhysicsSystem::IsAvailable() const {
#ifdef ENABLE_ML_FEATURES
    return m_modelLoaded;
#else
    return false;
#endif
}

void PINNPhysicsSystem::SetHybridThreshold(float radiusMultiplier) {
    m_hybridThresholdRadius = radiusMultiplier * R_ISCO;
    LOG_INFO("[PINN] Hybrid threshold set to {:.1f}× R_ISCO ({:.1f} units)", radiusMultiplier, m_hybridThresholdRadius);
}

// === Physics Parameter Controls (v2 model) ===

void PINNPhysicsSystem::SetBlackHoleMass(float normalized) {
    // Clamp to valid range (0.5 - 2.0)
    m_physicsParams.blackHoleMassNormalized = std::clamp(normalized, 0.5f, 2.0f);
    if (m_isV2Model) {
        LOG_INFO("[PINN] Black hole mass set to {:.2f}× default", m_physicsParams.blackHoleMassNormalized);
    }
}

void PINNPhysicsSystem::SetAlphaViscosity(float alpha) {
    // Clamp to valid range (0.01 - 0.3)
    m_physicsParams.alphaViscosity = std::clamp(alpha, 0.01f, 0.3f);
    if (m_isV2Model) {
        LOG_INFO("[PINN] Alpha viscosity set to {:.3f}", m_physicsParams.alphaViscosity);
    }
}

void PINNPhysicsSystem::SetDiskThickness(float hrRatio) {
    // Clamp to valid range (0.05 - 0.2)
    m_physicsParams.diskThickness = std::clamp(hrRatio, 0.05f, 0.2f);
    if (m_isV2Model) {
        LOG_INFO("[PINN] Disk thickness (H/R) set to {:.3f}", m_physicsParams.diskThickness);
    }
}

bool PINNPhysicsSystem::PredictForcesBatch(
    const DirectX::XMFLOAT3* positions,
    const DirectX::XMFLOAT3* velocities,
    DirectX::XMFLOAT3* outForces,
    uint32_t particleCount,
    float currentTime) {

#ifndef ENABLE_ML_FEATURES
    LOG_WARN("[PINN] ONNX Runtime not available");
    return false;
#else
    if (!m_enabled || !m_modelLoaded) {
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        // Prepare input tensor data: [particleCount, 7] for v1/v2 or [particleCount, 10] for v3
        int inputFeaturesPerParticle = m_isV3Model ? 10 : 7;
        std::vector<float> inputData;
        inputData.reserve(particleCount * inputFeaturesPerParticle);

        std::vector<uint32_t> pinnIndices;  // Indices of particles to process with PINN
        pinnIndices.reserve(particleCount);

        for (uint32_t i = 0; i < particleCount; i++) {
            // For hybrid mode radius check, convert to spherical (regardless of model version)
            float r_check = sqrtf(positions[i].x * positions[i].x +
                                 positions[i].y * positions[i].y +
                                 positions[i].z * positions[i].z);

            // Hybrid mode: only use PINN for particles beyond threshold
            if (m_hybridMode && !ShouldUsePINN(r_check)) {
                // Mark for GPU shader processing (zero forces for now)
                outForces[i] = XMFLOAT3(0.0f, 0.0f, 0.0f);
                continue;
            }

            // Add to PINN batch
            pinnIndices.push_back(i);

            if (m_isV3Model) {
                // v3: 10D Cartesian input (x, y, z, vx, vy, vz, t, M_bh, alpha, H_R)
                inputData.push_back(positions[i].x);
                inputData.push_back(positions[i].y);
                inputData.push_back(positions[i].z);
                inputData.push_back(velocities[i].x);
                inputData.push_back(velocities[i].y);
                inputData.push_back(velocities[i].z);
                inputData.push_back(currentTime);
                inputData.push_back(m_physicsParams.blackHoleMassNormalized);
                inputData.push_back(m_physicsParams.alphaViscosity);
                inputData.push_back(m_physicsParams.diskThickness);

                // DEBUG: Log first particle input
                if (i == 0) {
                    float r = sqrtf(positions[i].x * positions[i].x +
                                  positions[i].y * positions[i].y +
                                  positions[i].z * positions[i].z);
                    float v_mag = sqrtf(velocities[i].x * velocities[i].x +
                                      velocities[i].y * velocities[i].y +
                                      velocities[i].z * velocities[i].z);
                    LOG_INFO("[PINN v3 INPUT DEBUG] particle[0]: pos=({:.2f},{:.2f},{:.2f}) r={:.2f} | vel=({:.3f},{:.3f},{:.3f}) mag={:.3f} | t={:.3f} M_bh={:.3f} α={:.3f} H/R={:.3f}",
                             positions[i].x, positions[i].y, positions[i].z, r,
                             velocities[i].x, velocities[i].y, velocities[i].z, v_mag,
                             currentTime, m_physicsParams.blackHoleMassNormalized,
                             m_physicsParams.alphaViscosity, m_physicsParams.diskThickness);
                }
            } else {
                // v1/v2: 7D spherical input (r, θ, φ, v_r, v_θ, v_φ, t)
                ParticleStateSpherical state = CartesianToSpherical(positions[i], velocities[i]);
                inputData.push_back(state.r);
                inputData.push_back(state.theta);
                inputData.push_back(state.phi);
                inputData.push_back(state.v_r);
                inputData.push_back(state.v_theta);
                inputData.push_back(state.v_phi);
                inputData.push_back(currentTime);
            }
        }

        uint32_t pinnParticleCount = static_cast<uint32_t>(pinnIndices.size());

        if (pinnParticleCount == 0) {
            // No particles for PINN, all handled by GPU shader
            return true;
        }

        // Run ONNX inference
        std::vector<float> outputData;
        outputData.resize(pinnParticleCount * 3);

        // Update input shape for current batch
        std::vector<int64_t> inputShape = { static_cast<int64_t>(pinnParticleCount), inputFeaturesPerParticle };

        // Create state input tensor
        Ort::Value stateTensor = Ort::Value::CreateTensor<float>(
            m_memoryInfo,
            inputData.data(),
            inputData.size(),
            inputShape.data(),
            inputShape.size()
        );

        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(stateTensor));

        // For v2 model (NOT v3), create physics parameters tensor
        // v3 includes params in the main 10D input, so no separate tensor needed
        std::vector<float> paramsData;
        if (m_isV2Model && !m_isV3Model) {
            // Replicate parameters for each particle in batch
            paramsData.reserve(pinnParticleCount * 3);
            for (uint32_t i = 0; i < pinnParticleCount; i++) {
                paramsData.push_back(m_physicsParams.blackHoleMassNormalized);
                paramsData.push_back(m_physicsParams.alphaViscosity);
                paramsData.push_back(m_physicsParams.diskThickness);
            }

            std::vector<int64_t> paramsShape = { static_cast<int64_t>(pinnParticleCount), 3 };
            Ort::Value paramsTensor = Ort::Value::CreateTensor<float>(
                m_memoryInfo,
                paramsData.data(),
                paramsData.size(),
                paramsShape.data(),
                paramsShape.size()
            );
            inputTensors.push_back(std::move(paramsTensor));
        }

        // Run inference
        std::vector<const char*> inputNames;
        for (const auto& name : m_inputNames) {
            inputNames.push_back(name.c_str());
        }
        const char* outputNames[] = { m_outputNames[0].c_str() };

        auto outputTensors = m_ortSession->Run(
            Ort::RunOptions{ nullptr },
            inputNames.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNames,
            1
        );

        // Extract output
        float* outputPtr = outputTensors[0].GetTensorMutableData<float>();
        std::copy(outputPtr, outputPtr + pinnParticleCount * 3, outputData.begin());

        // CRITICAL: v3 outputs Cartesian forces (Fx, Fy, Fz) DIRECTLY
        // v1/v2 output spherical forces (F_r, F_theta, F_phi) that need conversion
        if (m_isV3Model) {
            // v3: Use raw ONNX output as Cartesian forces (NO coordinate transformation!)
            for (uint32_t i = 0; i < pinnParticleCount; i++) {
                uint32_t particleIdx = pinnIndices[i];

                outForces[particleIdx].x = outputData[i * 3 + 0];  // Fx (Cartesian)
                outForces[particleIdx].y = outputData[i * 3 + 1];  // Fy (Cartesian)
                outForces[particleIdx].z = outputData[i * 3 + 2];  // Fz (Cartesian)

                // DIAGNOSTIC: Log first particle to verify strong forces (GM=100 model)
                if (i == 0) {
                    float fx = outForces[particleIdx].x;
                    float fy = outForces[particleIdx].y;
                    float fz = outForces[particleIdx].z;
                    float mag = sqrtf(fx*fx + fy*fy + fz*fz);

                    // Compute radial component (should be NEGATIVE for attractive gravity)
                    float r = sqrtf(positions[particleIdx].x * positions[particleIdx].x +
                                  positions[particleIdx].y * positions[particleIdx].y +
                                  positions[particleIdx].z * positions[particleIdx].z);
                    float f_radial = (fx * positions[particleIdx].x +
                                     fy * positions[particleIdx].y +
                                     fz * positions[particleIdx].z) / r;

                    LOG_INFO("[PINN v3 DEBUG] RAW ONNX particle[0]: F=({:.6f}, {:.6f}, {:.6f}) mag={:.6f} | r={:.2f} F_radial={:.6f} (should be NEGATIVE!)",
                             fx, fy, fz, mag, r, f_radial);
                }
            }
        } else {
            // v1/v2: Convert spherical forces to Cartesian (legacy behavior)
            for (uint32_t i = 0; i < pinnParticleCount; i++) {
                uint32_t particleIdx = pinnIndices[i];

                PredictedForces forces;
                forces.F_r = outputData[i * 3 + 0];
                forces.F_theta = outputData[i * 3 + 1];
                forces.F_phi = outputData[i * 3 + 2];

                // Convert back to Cartesian
                ParticleStateSpherical state = CartesianToSpherical(positions[particleIdx], velocities[particleIdx]);
                outForces[particleIdx] = SphericalForcesToCartesian(forces, state);
            }
        }

        // Update performance metrics
        auto endTime = std::chrono::high_resolution_clock::now();
        float elapsedMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

        m_metrics.inferenceTimeMs = elapsedMs;
        m_metrics.particlesProcessed = pinnParticleCount;
        m_metrics.batchCount++;
        m_metrics.avgBatchTimeMs = (m_metrics.avgBatchTimeMs * (m_metrics.batchCount - 1) + elapsedMs) / m_metrics.batchCount;

        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("[PINN] Inference failed: {}", e.what());
        return false;
    }
#endif
}

bool PINNPhysicsSystem::PredictForces(
    const DirectX::XMFLOAT3& position,
    const DirectX::XMFLOAT3& velocity,
    DirectX::XMFLOAT3& outForce,
    float currentTime) {

    return PredictForcesBatch(&position, &velocity, &outForce, 1, currentTime);
}

void PINNPhysicsSystem::ResetMetrics() {
    m_metrics = {};
}

std::string PINNPhysicsSystem::GetModelInfo() const {
#ifndef ENABLE_ML_FEATURES
    return "ONNX Runtime not available (compiled without ENABLE_ML_FEATURES)";
#else
    if (!m_modelLoaded) {
        return "No model loaded";
    }

    std::ostringstream oss;
    oss << "PINN Accretion Disk Model (v" << (m_isV2Model ? "2" : "1") << ")\n";
    oss << "  Input 0: " << m_inputNames[0] << " [batch, 7]\n";
    if (m_isV2Model && m_inputNames.size() > 1) {
        oss << "  Input 1: " << m_inputNames[1] << " [batch, 3]\n";
    }
    oss << "  Output: " << m_outputNames[0] << " [batch, 3]\n";
    oss << "  Hybrid Mode: " << (m_hybridMode ? "ON" : "OFF") << "\n";
    oss << "  Threshold: " << (m_hybridThresholdRadius / R_ISCO) << "× R_ISCO\n";
    if (m_isV2Model) {
        oss << "  --- Physics Parameters ---\n";
        oss << "  Black Hole Mass: " << m_physicsParams.blackHoleMassNormalized << "× default\n";
        oss << "  Alpha Viscosity: " << m_physicsParams.alphaViscosity << "\n";
        oss << "  Disk Thickness: " << m_physicsParams.diskThickness << " (H/R)\n";
    }
    oss << "  Status: " << (m_enabled ? "ENABLED" : "DISABLED");

    return oss.str();
#endif
}

// === Private Methods ===

PINNPhysicsSystem::ParticleStateSpherical PINNPhysicsSystem::CartesianToSpherical(
    const DirectX::XMFLOAT3& position,
    const DirectX::XMFLOAT3& velocity) const {

    ParticleStateSpherical state;

    // Position in spherical coordinates
    float x = position.x;
    float y = position.y;
    float z = position.z;

    state.r = std::sqrt(x * x + y * y + z * z);
    state.theta = std::acos(z / (state.r + 1e-6f));  // Avoid division by zero
    state.phi = std::atan2(y, x);

    // Velocity in spherical coordinates (using Jacobian transformation)
    float sin_theta = std::sin(state.theta);
    float cos_theta = std::cos(state.theta);
    float sin_phi = std::sin(state.phi);
    float cos_phi = std::cos(state.phi);

    // Transform velocity components
    // v_r = v · r̂
    // v_θ = v · θ̂
    // v_φ = v · φ̂
    state.v_r = (x * velocity.x + y * velocity.y + z * velocity.z) / (state.r + 1e-6f);
    state.v_theta = (cos_theta * cos_phi * velocity.x + cos_theta * sin_phi * velocity.y - sin_theta * velocity.z);
    state.v_phi = (-sin_phi * velocity.x + cos_phi * velocity.y);

    return state;
}

DirectX::XMFLOAT3 PINNPhysicsSystem::SphericalForcesToCartesian(
    const PredictedForces& forces,
    const ParticleStateSpherical& state) const {

    float sin_theta = std::sin(state.theta);
    float cos_theta = std::cos(state.theta);
    float sin_phi = std::sin(state.phi);
    float cos_phi = std::cos(state.phi);

    // Spherical unit vectors in Cartesian coordinates:
    // r̂ = (sin θ cos φ, sin θ sin φ, cos θ)
    // θ̂ = (cos θ cos φ, cos θ sin φ, -sin θ)
    // φ̂ = (-sin φ, cos φ, 0)

    float F_x = forces.F_r * sin_theta * cos_phi +
                forces.F_theta * cos_theta * cos_phi +
                forces.F_phi * (-sin_phi);

    float F_y = forces.F_r * sin_theta * sin_phi +
                forces.F_theta * cos_theta * sin_phi +
                forces.F_phi * cos_phi;

    float F_z = forces.F_r * cos_theta +
                forces.F_theta * (-sin_theta);

    return XMFLOAT3(F_x, F_y, F_z);
}

bool PINNPhysicsSystem::ShouldUsePINN(float radius) const {
    if (!m_hybridMode) {
        return true;  // Always use PINN if not in hybrid mode
    }

    // Use PINN only for particles beyond hybrid threshold
    // Near ISCO, use GPU shader for more accurate physics
    return radius > m_hybridThresholdRadius;
}
