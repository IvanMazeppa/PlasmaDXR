"""
Struct Validator Tool
Validates proposed particle structures for GPU compatibility
"""

import re


async def validate_particle_struct(
    struct_definition: str,
    check_backward_compatibility: bool = True,
    check_gpu_alignment: bool = True
) -> str:
    """
    Validate particle structure definition

    Args:
        struct_definition: C++ struct code to validate
        check_backward_compatibility: Check against 32-byte legacy format
        check_gpu_alignment: Validate 16-byte GPU alignment

    Returns:
        Validation report with pass/fail for each check
    """

    report = "# Particle Structure Validation\n\n"

    # Display the struct being validated
    report += "## Struct Definition\n\n"
    report += "```cpp\n"
    report += struct_definition
    report += "\n```\n\n"

    # Parse struct fields
    report += "## Field Analysis\n\n"

    fields = []
    field_pattern = r'(XMFLOAT\d+|float|int|uint32_t|uint64_t|DirectX::XMFLOAT\d+)\s+(\w+);'

    for match in re.finditer(field_pattern, struct_definition):
        field_type = match.group(1).replace("DirectX::", "")
        field_name = match.group(2)
        fields.append((field_type, field_name))

    if not fields:
        report += "⚠️  Warning: Could not parse struct fields. Manual validation required.\n\n"
        return report

    report += "| Field | Type | Size (bytes) | Offset |\n"
    report += "|-------|------|--------------|--------|\n"

    # Calculate sizes and offsets
    type_sizes = {
        "XMFLOAT2": 8,
        "XMFLOAT3": 12,
        "XMFLOAT4": 16,
        "float": 4,
        "int": 4,
        "uint32_t": 4,
        "uint64_t": 8
    }

    current_offset = 0
    total_size = 0
    legacy_fields = ["position", "velocity", "temperature", "radius", "lifetime"]

    for field_type, field_name in fields:
        size = type_sizes.get(field_type, 4)
        is_legacy = field_name in legacy_fields

        legacy_marker = " (Legacy)" if is_legacy else " (NEW)"
        report += f"| {field_name} | {field_type} | {size} | {current_offset}{legacy_marker} |\n"

        current_offset += size
        total_size += size

    report += f"\n**Total Size:** {total_size} bytes\n\n"

    # GPU Alignment Check
    if check_gpu_alignment:
        report += "## GPU Alignment Validation\n\n"

        if total_size % 16 == 0:
            report += f"✅ **PASS**: Structure is 16-byte aligned ({total_size} bytes)\n"
        else:
            padding_needed = 16 - (total_size % 16)
            padded_size = total_size + padding_needed
            report += f"❌ **FAIL**: Structure is NOT 16-byte aligned ({total_size} bytes)\n"
            report += f"   - Current size: {total_size} bytes\n"
            report += f"   - Padding needed: {padding_needed} bytes\n"
            report += f"   - Aligned size: {padded_size} bytes\n\n"

            report += "**Recommendation:** Add padding field:\n"
            report += "```cpp\n"
            if padding_needed == 4:
                report += "float _padding;\n"
            elif padding_needed == 8:
                report += "float _padding[2];\n"
            elif padding_needed == 12:
                report += "float _padding[3];\n"
            report += "```\n"

        report += "\n"

    # Backward Compatibility Check
    if check_backward_compatibility:
        report += "## Backward Compatibility Validation\n\n"

        legacy_check_results = []

        # Check if legacy fields exist and are in correct positions
        expected_legacy_layout = [
            ("position", "XMFLOAT3", 0),
            ("velocity", "XMFLOAT3", 12),
            ("temperature", "float", 24),
            ("radius", "float", 28)
        ]

        actual_layout = {name: (ftype, i) for i, (ftype, name) in enumerate(fields)}

        report += "**Legacy Field Validation:**\n\n"

        all_legacy_match = True
        current_offset = 0

        for expected_name, expected_type, expected_offset in expected_legacy_layout:
            if expected_name in actual_layout:
                actual_type, actual_index = actual_layout[expected_name]

                # Calculate actual offset
                actual_offset = 0
                for i in range(actual_index):
                    actual_offset += type_sizes.get(fields[i][0], 4)

                if actual_type == expected_type and actual_offset == expected_offset:
                    report += f"✅ `{expected_name}` - Correct type ({expected_type}) and offset ({expected_offset})\n"
                else:
                    report += f"❌ `{expected_name}` - Type or offset mismatch\n"
                    report += f"   - Expected: {expected_type} @ offset {expected_offset}\n"
                    report += f"   - Actual: {actual_type} @ offset {actual_offset}\n"
                    all_legacy_match = False
            else:
                report += f"❌ `{expected_name}` - Missing field\n"
                all_legacy_match = False

        report += "\n"

        if all_legacy_match and total_size >= 32:
            report += "✅ **PASS**: Backward compatible with 32-byte legacy format\n"
            report += "   - All legacy fields present in correct positions\n"
            report += "   - New fields appended after legacy data\n"
        else:
            report += "❌ **FAIL**: NOT backward compatible with legacy format\n"
            report += "   - Legacy code will break or read incorrect data\n"
            report += "   - Requires full system update (no incremental rollout)\n"

        report += "\n"

    # Size Category Analysis
    report += "## Size Category Analysis\n\n"

    if total_size <= 32:
        category = "Baseline"
        overhead = "0%"
        status = "✅"
    elif total_size <= 48:
        category = "Minimal Extension"
        overhead = f"+{((total_size - 32) / 32 * 100):.0f}%"
        status = "✅"
    elif total_size <= 64:
        category = "Full Extension"
        overhead = f"+{((total_size - 32) / 32 * 100):.0f}%"
        status = "⚠️"
    else:
        category = "Oversized"
        overhead = f"+{((total_size - 32) / 32 * 100):.0f}%"
        status = "❌"

    report += f"**Category:** {category} {status}\n"
    report += f"**Size:** {total_size} bytes\n"
    report += f"**Overhead:** {overhead} vs 32-byte baseline\n\n"

    report += "**Memory Usage Impact:**\n"
    report += f"- @ 10K particles: {(total_size * 10000) / (1024 * 1024):.2f} MB\n"
    report += f"- @ 100K particles: {(total_size * 100000) / (1024 * 1024):.2f} MB\n\n"

    # Common Issues Check
    report += "## Common Issues Check\n\n"

    issues_found = []

    # Check for float3 alignment issues (should be XMFLOAT3)
    if "float3" in struct_definition:
        issues_found.append("⚠️  Using HLSL `float3` instead of C++ `XMFLOAT3` - causes alignment mismatch")

    # Check for bool types
    if "bool" in struct_definition.lower():
        issues_found.append("⚠️  Using `bool` type - use `uint32_t` for GPU compatibility (bool is 1 byte in C++, 4 bytes in HLSL)")

    # Check for double types
    if "double" in struct_definition.lower():
        issues_found.append("⚠️  Using `double` type - GPUs prefer `float` for performance")

    # Check for arrays without explicit size
    if re.search(r'\w+\s+\w+\[\];', struct_definition):
        issues_found.append("❌ Flexible array member detected - not allowed in GPU structures")

    # Check for std:: types
    if "std::" in struct_definition:
        issues_found.append("❌ STL types detected - not GPU-compatible (use plain arrays or DirectX types)")

    if issues_found:
        for issue in issues_found:
            report += f"{issue}\n"
    else:
        report += "✅ No common issues detected\n"

    report += "\n"

    # Final Verdict
    report += "## Final Validation Result\n\n"

    all_checks_pass = True
    if check_gpu_alignment and total_size % 16 != 0:
        all_checks_pass = False
    if check_backward_compatibility and not all_legacy_match:
        all_checks_pass = False
    if issues_found:
        all_checks_pass = False

    if all_checks_pass:
        report += "✅ **VALIDATION PASSED**\n\n"
        report += "This structure is ready for GPU use:\n"
        report += f"- 16-byte aligned: ✅\n"
        if check_backward_compatibility:
            report += "- Backward compatible: ✅\n"
        report += "- No critical issues: ✅\n\n"
        report += "**Recommendation:** Proceed with implementation\n"

    else:
        report += "❌ **VALIDATION FAILED**\n\n"
        report += "This structure requires corrections before GPU use:\n"

        if check_gpu_alignment and total_size % 16 != 0:
            report += "- 16-byte alignment: ❌ (add padding)\n"
        if check_backward_compatibility and not all_legacy_match:
            report += "- Backward compatibility: ❌ (reorder fields)\n"
        if issues_found:
            report += f"- Critical issues: ❌ ({len(issues_found)} found)\n"

        report += "\n**Recommendation:** Fix issues before implementation\n"

    report += "\n---\n"
    report += f"**Validation completed: {total_size} bytes, {len(fields)} fields**\n"

    return report
