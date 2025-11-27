#include "SIRENVortexField.h"
#include "../utils/Logger.h"
#include <cmath>
#include <chrono>
#include <sstream>
#include <filesystem>

using namespace DirectX;
namespace fs = std::filesystem;

SIRENVortexField::SIRENVortexField()
#ifdef ENABLE_ML_FEATURES
    : m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
    LOG_INFO("[SIREN] Initializing SIREN Vortex Field system...");
}

SIRENVortexField::~SIRENVortexField() {
#ifdef ENABLE_ML_FEATURES
    m_ortSession.reset();
    m_sessionOptions.reset();
    m_ortEnv.reset();
#endif
    LOG_INFO("[SIREN] Shutting down SIREN vortex field system");
}

bool SIRENVortexField::Initialize(const std::string& modelPath) {
#ifndef ENABLE_ML_FEATURES
    LOG_WARN("[SIREN] ONNX Runtime not available. SIREN features disabled.");
    return false;
#else
    try {
        LOG_INFO("[SIREN] Loading SIREN vortex model from: {}", modelPath);

        // Create ONNX Runtime environment
        m_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PlasmaDX-SIREN");

        // Configure session options
        m_sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_sessionOptions->SetIntraOpNumThreads(8);  // Smaller model, fewer threads
        m_sessionOptions->SetInterOpNumThreads(2);
        m_sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        LOG_INFO("[SIREN] ONNX Runtime configured: 8 intra-op threads, 2 inter-op threads");

        // Convert path to wide string (Windows requirement)
        std::wstring wideModelPath(modelPath.begin(), modelPath.end());

        // Create inference session
        m_ortSession = std::make_unique<Ort::Session>(*m_ortEnv, wideModelPath.c_str(), *m_sessionOptions);

        // Get input/output metadata
        Ort::AllocatorWithDefaultOptions allocator;

        // Input info (should be [batch, 5])
        size_t numInputs = m_ortSession->GetInputCount();
        if (numInputs != 1) {
            LOG_ERROR("[SIREN] Expected 1 input, found {}", numInputs);
            return false;
        }

        auto inputName = m_ortSession->GetInputNameAllocated(0, allocator);
        m_inputNames.push_back(std::string(inputName.get()));

        auto inputTypeInfo = m_ortSession->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        m_inputShape = inputTensorInfo.GetShape();

        LOG_INFO("[SIREN] Input: '{}', shape: [{}, {}]", m_inputNames[0], m_inputShape[0], m_inputShape[1]);

        // Validate input dimensions
        if (m_inputShape[1] != 5) {
            LOG_ERROR("[SIREN] Invalid input shape. Expected 5 (x,y,z,t,seed), got {}", m_inputShape[1]);
            return false;
        }

        // Output info (should be [batch, 3])
        size_t numOutputs = m_ortSession->GetOutputCount();
        if (numOutputs != 1) {
            LOG_ERROR("[SIREN] Expected 1 output, found {}", numOutputs);
            return false;
        }

        auto outputName = m_ortSession->GetOutputNameAllocated(0, allocator);
        m_outputNames.push_back(std::string(outputName.get()));

        auto outputTypeInfo = m_ortSession->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        m_outputShape = outputTensorInfo.GetShape();

        LOG_INFO("[SIREN] Output: '{}', shape: [{}, {}]", m_outputNames[0], m_outputShape[0], m_outputShape[1]);

        if (m_outputShape[1] != 3) {
            LOG_ERROR("[SIREN] Invalid output shape. Expected 3 (vorticity), got {}", m_outputShape[1]);
            return false;
        }

        // Store model info
        fs::path p(modelPath);
        m_modelName = p.stem().string();
        m_modelPath = modelPath;
        m_modelLoaded = true;

        LOG_INFO("[SIREN] Successfully loaded SIREN vortex field model!");
        LOG_INFO("[SIREN] Intensity: {:.2f}, Seed: {:.2f}", m_intensity, m_seed);

        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("[SIREN] ONNX Runtime exception: {}", e.what());
        return false;
    }
#endif
}

bool SIRENVortexField::IsAvailable() const {
#ifdef ENABLE_ML_FEATURES
    return m_modelLoaded;
#else
    return false;
#endif
}

bool SIRENVortexField::PredictVorticityBatch(
    const DirectX::XMFLOAT3* positions,
    DirectX::XMFLOAT3* outVorticity,
    uint32_t particleCount,
    float currentTime) {

#ifndef ENABLE_ML_FEATURES
    LOG_WARN("[SIREN] ONNX Runtime not available");
    return false;
#else
    if (!m_enabled || !m_modelLoaded) {
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        // Prepare input tensor: [particleCount, 5] = [x, y, z, t, seed]
        std::vector<float> inputData;
        inputData.reserve(particleCount * 5);

        for (uint32_t i = 0; i < particleCount; i++) {
            inputData.push_back(positions[i].x);
            inputData.push_back(positions[i].y);
            inputData.push_back(positions[i].z);
            inputData.push_back(currentTime);
            inputData.push_back(m_seed);  // Same seed for all particles (coherent field)
        }

        // Create input tensor
        std::vector<int64_t> inputShape = { static_cast<int64_t>(particleCount), 5 };
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            m_memoryInfo,
            inputData.data(),
            inputData.size(),
            inputShape.data(),
            inputShape.size()
        );

        // Run inference
        const char* inputNames[] = { m_inputNames[0].c_str() };
        const char* outputNames[] = { m_outputNames[0].c_str() };

        auto outputTensors = m_ortSession->Run(
            Ort::RunOptions{ nullptr },
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );

        // Extract output
        float* outputPtr = outputTensors[0].GetTensorMutableData<float>();

        // Copy to output with intensity scaling
        for (uint32_t i = 0; i < particleCount; i++) {
            outVorticity[i].x = outputPtr[i * 3 + 0] * m_intensity;
            outVorticity[i].y = outputPtr[i * 3 + 1] * m_intensity;
            outVorticity[i].z = outputPtr[i * 3 + 2] * m_intensity;
        }

        // Update performance metrics
        auto endTime = std::chrono::high_resolution_clock::now();
        float elapsedMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();

        m_metrics.inferenceTimeMs = elapsedMs;
        m_metrics.particlesProcessed = particleCount;
        m_metrics.batchCount++;
        m_metrics.avgBatchTimeMs = (m_metrics.avgBatchTimeMs * (m_metrics.batchCount - 1) + elapsedMs) / m_metrics.batchCount;

        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("[SIREN] Inference failed: {}", e.what());
        return false;
    }
#endif
}

bool SIRENVortexField::ComputeTurbulentForcesBatch(
    const DirectX::XMFLOAT3* positions,
    const DirectX::XMFLOAT3* velocities,
    DirectX::XMFLOAT3* outForces,
    uint32_t particleCount,
    float currentTime) {

    // First get vorticity
    std::vector<XMFLOAT3> vorticity(particleCount);
    if (!PredictVorticityBatch(positions, vorticity.data(), particleCount, currentTime)) {
        // Fill with zeros on failure
        for (uint32_t i = 0; i < particleCount; i++) {
            outForces[i] = XMFLOAT3(0.0f, 0.0f, 0.0f);
        }
        return false;
    }

    // Convert vorticity to force: F = cross(velocity, vorticity)
    // This creates a rotating force perpendicular to both velocity and the vorticity axis
    for (uint32_t i = 0; i < particleCount; i++) {
        const XMFLOAT3& v = velocities[i];
        const XMFLOAT3& w = vorticity[i];

        // Cross product: v × ω
        outForces[i].x = v.y * w.z - v.z * w.y;
        outForces[i].y = v.z * w.x - v.x * w.z;
        outForces[i].z = v.x * w.y - v.y * w.x;
    }

    return true;
}

bool SIRENVortexField::PredictVorticity(
    const DirectX::XMFLOAT3& position,
    DirectX::XMFLOAT3& outVorticity,
    float currentTime) {

    return PredictVorticityBatch(&position, &outVorticity, 1, currentTime);
}

void SIRENVortexField::ResetMetrics() {
    m_metrics = {};
}

std::string SIRENVortexField::GetModelInfo() const {
#ifndef ENABLE_ML_FEATURES
    return "ONNX Runtime not available";
#else
    if (!m_modelLoaded) {
        return "No model loaded";
    }

    std::ostringstream oss;
    oss << "SIREN Vortex Field Model\n";
    oss << "  Name: " << m_modelName << "\n";
    oss << "  Input: [batch, 5] (x, y, z, t, seed)\n";
    oss << "  Output: [batch, 3] (vorticity)\n";
    oss << "  Intensity: " << m_intensity << "\n";
    oss << "  Seed: " << m_seed << "\n";
    oss << "  Status: " << (m_enabled ? "ENABLED" : "DISABLED");

    return oss.str();
#endif
}

