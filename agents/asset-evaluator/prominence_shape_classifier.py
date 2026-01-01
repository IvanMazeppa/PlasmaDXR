"""
Phase 4: Prominence Shape Classifier

Detects morphological anomalies in solar prominences by analyzing:
1. Contour shape characteristics (curvature, symmetry, aspect ratio)
2. Filamentary structure presence
3. Natural vs artificial edge patterns
4. "Cat ear" and other synthetic artifacts

Real solar prominences have:
- Irregular, asymmetric shapes
- Filamentary/thread-like internal structure
- Smooth, flowing curvature
- Variable brightness along length

Synthetic artifacts often show:
- Symmetric "cat ear" triangular protrusions
- Regular ribbon-like shapes
- Sharp angular transitions
- Uniform brightness/texture
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ProminenceType(Enum):
    """Classification of prominence shapes."""
    NATURAL_LOOP = "natural_loop"           # Realistic arching prominence
    NATURAL_ERUPTION = "natural_eruption"   # Realistic eruptive prominence
    NATURAL_FILAMENT = "natural_filament"   # Realistic filamentary structure
    CAT_EAR = "cat_ear"                     # Symmetric triangular artifact
    RIBBON = "ribbon"                        # Unnatural ribbon shape
    BLOB = "blob"                           # Amorphous unstructured mass
    ANGULAR = "angular"                     # Sharp unnatural angles
    UNKNOWN = "unknown"


@dataclass
class ProminenceRegion:
    """Represents a detected prominence region."""
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    centroid: Tuple[float, float]
    angle_from_disk_center: float  # Radial angle from sun center
    distance_from_limb: float      # Distance from solar limb

    # Shape characteristics
    aspect_ratio: float
    solidity: float                # Area / convex hull area
    symmetry_score: float          # 0=asymmetric (natural), 1=symmetric (suspicious)
    curvature_variance: float      # High = natural, Low = artificial
    max_curvature: float
    min_curvature: float

    # Texture characteristics
    internal_variance: float       # High = filamentary, Low = uniform
    edge_sharpness: float

    # Classification
    prominence_type: ProminenceType
    confidence: float
    issues: List[str]


class ProminenceShapeClassifier:
    """
    Analyzes solar prominence morphology to detect synthetic artifacts.

    Key detection targets:
    - "Cat ear" triangular protrusions (symmetric, angular)
    - Ribbon artifacts (too regular, uniform width)
    - Missing filamentary structure
    - Unnatural symmetry
    """

    def __init__(self):
        # Thresholds for anomaly detection
        self.symmetry_threshold = 0.7       # Above = suspiciously symmetric
        self.curvature_var_threshold = 0.1  # Below = suspiciously uniform
        self.aspect_ratio_ribbon = 5.0      # Above = ribbon-like
        self.solidity_blob = 0.9            # Above = blob-like (no structure)
        self.internal_var_threshold = 0.15  # Below = lacks filamentary texture

        # Shape templates for known artifacts
        self.cat_ear_template = self._create_cat_ear_template()

    def _create_cat_ear_template(self) -> np.ndarray:
        """Create template for cat ear detection via template matching."""
        # Triangular "cat ear" shape
        template = np.zeros((50, 40), dtype=np.uint8)
        pts = np.array([[20, 0], [0, 50], [40, 50]], dtype=np.int32)
        cv2.fillPoly(template, [pts], 255)
        return template

    def detect_solar_disk(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """
        Detect the solar disk center and radius.

        Returns:
            (center_xy, radius) or (None, None) if not detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Threshold to find bright disk
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find largest contour (should be the solar disk)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)

        # Fit minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest)

        return (int(cx), int(cy)), int(radius)

    def extract_prominence_mask(self, image: np.ndarray,
                                  disk_center: Tuple[int, int],
                                  disk_radius: int) -> np.ndarray:
        """
        Extract mask of prominence regions (bright areas beyond the solar limb).

        Args:
            image: RGB image
            disk_center: (x, y) center of solar disk
            disk_radius: Radius of solar disk

        Returns:
            Binary mask of prominence regions
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Create disk mask (slightly inside the limb)
        disk_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(disk_mask, disk_center, int(disk_radius * 0.98), 255, -1)

        # Create corona region mask (beyond limb but within detection range)
        corona_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(corona_mask, disk_center, int(disk_radius * 1.5), 255, -1)
        cv2.circle(corona_mask, disk_center, int(disk_radius * 0.98), 0, -1)

        # Find bright regions in corona
        # Use adaptive threshold for prominence detection
        prominence_thresh = np.percentile(gray[corona_mask > 0], 70) if np.any(corona_mask > 0) else 30
        _, bright_mask = cv2.threshold(gray, prominence_thresh, 255, cv2.THRESH_BINARY)

        # Prominence mask = bright regions in corona zone
        prominence_mask = cv2.bitwise_and(bright_mask, corona_mask)

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        prominence_mask = cv2.morphologyEx(prominence_mask, cv2.MORPH_CLOSE, kernel)
        prominence_mask = cv2.morphologyEx(prominence_mask, cv2.MORPH_OPEN, kernel)

        return prominence_mask

    def analyze_contour_curvature(self, contour: np.ndarray) -> Dict[str, float]:
        """
        Analyze the curvature profile of a contour.

        Real prominences have smooth, varying curvature.
        "Cat ears" have sharp angular transitions.
        """
        if len(contour) < 10:
            return {"mean": 0, "variance": 0, "max": 0, "min": 0, "angular_count": 0}

        # Smooth contour for better curvature estimation
        contour = contour.reshape(-1, 2).astype(np.float32)

        # Compute curvature at each point using finite differences
        curvatures = []
        n = len(contour)

        for i in range(n):
            # Get neighboring points
            p_prev = contour[(i - 2) % n]
            p_curr = contour[i]
            p_next = contour[(i + 2) % n]

            # Vectors
            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            # Cross product for curvature direction, magnitude for curvature amount
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)

            if mag1 > 0 and mag2 > 0:
                # Curvature approximation
                curvature = cross / (mag1 * mag2 + 1e-6)
                curvatures.append(curvature)

        if not curvatures:
            return {"mean": 0, "variance": 0, "max": 0, "min": 0, "angular_count": 0}

        curvatures = np.array(curvatures)

        # Count angular transitions (sharp curvature changes)
        curvature_diff = np.abs(np.diff(curvatures))
        angular_count = np.sum(curvature_diff > 0.3)  # Threshold for "sharp" turn

        return {
            "mean": float(np.mean(np.abs(curvatures))),
            "variance": float(np.var(curvatures)),
            "max": float(np.max(curvatures)),
            "min": float(np.min(curvatures)),
            "angular_count": int(angular_count)
        }

    def compute_symmetry(self, contour: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Compute bilateral symmetry score.

        Real prominences are asymmetric (low score).
        "Cat ears" are symmetric (high score).

        Returns:
            Symmetry score 0-1 (1 = perfectly symmetric)
        """
        x, y, w, h = bbox
        if w < 5 or h < 5:
            return 0.0

        # Create binary mask from contour
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour.reshape(-1, 2) - np.array([x, y])
        shifted_contour = shifted_contour.astype(np.int32)
        cv2.fillPoly(mask, [shifted_contour], 255)

        # Compute vertical symmetry
        mid = w // 2
        left_half = mask[:, :mid]
        right_half = cv2.flip(mask[:, -mid:], 1)

        # Resize to same shape if needed
        if left_half.shape != right_half.shape:
            min_w = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_w]
            right_half = right_half[:, :min_w]

        # Compute overlap
        intersection = np.sum((left_half > 0) & (right_half > 0))
        union = np.sum((left_half > 0) | (right_half > 0))

        vertical_symmetry = intersection / (union + 1e-6)

        # Compute horizontal symmetry
        mid_h = h // 2
        top_half = mask[:mid_h, :]
        bottom_half = cv2.flip(mask[-mid_h:, :], 0)

        if top_half.shape != bottom_half.shape:
            min_h = min(top_half.shape[0], bottom_half.shape[0])
            top_half = top_half[:min_h, :]
            bottom_half = bottom_half[:min_h, :]

        intersection_h = np.sum((top_half > 0) & (bottom_half > 0))
        union_h = np.sum((top_half > 0) | (bottom_half > 0))

        horizontal_symmetry = intersection_h / (union_h + 1e-6)

        # Return max symmetry (either axis)
        return max(vertical_symmetry, horizontal_symmetry)

    def detect_cat_ear_pattern(self, contour: np.ndarray,
                                 bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Specifically detect "cat ear" triangular prominence artifacts.

        Cat ears are characterized by:
        - High symmetry
        - Triangular shape (high aspect ratio at tip)
        - Sharp apex angle
        - Low internal texture variance

        Returns:
            (is_cat_ear, confidence)
        """
        x, y, w, h = bbox
        if w < 10 or h < 10:
            return False, 0.0

        # Compute convex hull and check triangular nature
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)

        if hull_area == 0:
            return False, 0.0

        # Solidity check - triangles have high solidity
        solidity = contour_area / hull_area

        # Fit a triangle and check fit quality
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        is_triangular = len(approx) == 3

        # Check for sharp apex (characteristic of cat ears)
        if len(approx) >= 3:
            # Find the topmost point
            points = approx.reshape(-1, 2)
            apex_idx = np.argmin(points[:, 1])  # Topmost
            apex = points[apex_idx]

            # Compute angle at apex
            other_points = np.delete(points, apex_idx, axis=0)
            if len(other_points) >= 2:
                v1 = other_points[0] - apex
                v2 = other_points[1] - apex

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                apex_angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

                # Cat ears typically have acute apex angles (30-70 degrees)
                is_acute_apex = 30 < apex_angle < 70
            else:
                is_acute_apex = False
        else:
            is_acute_apex = False

        # Template matching with cat ear shape
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour.reshape(-1, 2) - np.array([x, y])
        cv2.fillPoly(mask, [shifted_contour.astype(np.int32)], 255)

        # Resize template to match
        template_resized = cv2.resize(self.cat_ear_template, (w, h))

        # Compute similarity
        if np.sum(mask) > 0 and np.sum(template_resized) > 0:
            intersection = np.sum((mask > 0) & (template_resized > 0))
            union = np.sum((mask > 0) | (template_resized > 0))
            template_match = intersection / (union + 1e-6)
        else:
            template_match = 0

        # Combine signals
        cat_ear_score = 0.0
        if is_triangular:
            cat_ear_score += 0.3
        if is_acute_apex:
            cat_ear_score += 0.25
        if solidity > 0.85:
            cat_ear_score += 0.2
        if template_match > 0.5:
            cat_ear_score += 0.25

        is_cat_ear = cat_ear_score > 0.6

        return is_cat_ear, cat_ear_score

    def analyze_internal_texture(self, image: np.ndarray,
                                   contour: np.ndarray,
                                   bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Analyze texture within the prominence region.

        Real prominences have filamentary internal structure.
        Synthetic artifacts often have uniform texture.
        """
        x, y, w, h = bbox
        if w < 5 or h < 5:
            return {"variance": 0, "edge_density": 0, "filamentary_score": 0}

        # Extract region
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        region = gray[y:y+h, x:x+w]

        # Create mask for this prominence
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour.reshape(-1, 2) - np.array([x, y])
        cv2.fillPoly(mask, [shifted_contour.astype(np.int32)], 255)

        # Compute variance within region
        masked_pixels = region[mask > 0]
        if len(masked_pixels) < 10:
            return {"variance": 0, "edge_density": 0, "filamentary_score": 0}

        variance = float(np.var(masked_pixels) / (np.mean(masked_pixels) + 1e-6))

        # Edge density within region (filamentary structure shows edges)
        edges = cv2.Canny(region, 50, 150)
        edge_density = float(np.sum(edges[mask > 0] > 0) / (np.sum(mask > 0) + 1e-6))

        # Filamentary score: high variance + moderate edge density
        filamentary_score = min(1.0, variance * 2 + edge_density * 5)

        return {
            "variance": variance,
            "edge_density": edge_density,
            "filamentary_score": filamentary_score
        }

    def classify_prominence(self, region: Dict) -> Tuple[ProminenceType, float, List[str]]:
        """
        Classify a prominence region based on its characteristics.

        Key insight: Real prominences have HIGH internal texture variance (filamentary)
        while synthetic artifacts have LOW variance (uniform procedural noise).

        Returns:
            (prominence_type, confidence, issues_list)
        """
        issues = []

        symmetry = region["symmetry"]
        curvature_var = region["curvature"]["variance"]
        aspect_ratio = region["aspect_ratio"]
        solidity = region["solidity"]
        internal_var = region["texture"]["variance"]
        edge_density = region["texture"]["edge_density"]
        angular_count = region["curvature"]["angular_count"]
        area = region.get("area", 100)
        is_cat_ear = region.get("is_cat_ear", False)
        cat_ear_confidence = region.get("cat_ear_confidence", 0.0)

        # CRITICAL: Internal texture variance is the KEY discriminator
        # Real prominences: internal_var > 2.0 (filamentary structure)
        # Synthetic artifacts: internal_var < 1.5 (uniform procedural)
        has_real_texture = internal_var > 2.0
        has_synthetic_texture = internal_var < 1.5

        # Normalize angular count by area (large regions naturally have more corners)
        normalized_angular = angular_count / max(np.sqrt(area), 10)

        # Check for cat ear pattern first (most specific)
        # But only if texture is synthetic-looking
        if is_cat_ear and has_synthetic_texture:
            issues.append(f"CAT_EAR_PATTERN: Triangular symmetric protrusion detected (conf={cat_ear_confidence:.2f})")
            return ProminenceType.CAT_EAR, cat_ear_confidence, issues

        # Check for ribbon artifact - only flag if texture is uniform
        if aspect_ratio > self.aspect_ratio_ribbon and has_synthetic_texture:
            if symmetry > 0.5:
                issues.append(f"RIBBON_ARTIFACT: Unnaturally regular ribbon shape (aspect={aspect_ratio:.1f}, symmetry={symmetry:.2f})")
                return ProminenceType.RIBBON, 0.8, issues

        # High internal variance = real filamentary structure
        if has_real_texture:
            # This is likely a real prominence with complex internal structure
            if aspect_ratio > 3:
                return ProminenceType.NATURAL_FILAMENT, 0.90, []
            elif region["curvature"]["mean"] > 0.2:
                return ProminenceType.NATURAL_LOOP, 0.90, []
            else:
                return ProminenceType.NATURAL_ERUPTION, 0.85, []

        # Synthetic texture + other indicators = artifact
        if has_synthetic_texture:
            # Check for unnatural symmetry
            if symmetry > self.symmetry_threshold:
                issues.append(f"HIGH_SYMMETRY: Suspicious bilateral symmetry ({symmetry:.2f} > {self.symmetry_threshold})")

            # Check for lack of curvature variation (too smooth/regular)
            if curvature_var < self.curvature_var_threshold:
                issues.append(f"UNIFORM_CURVATURE: Unnaturally regular curvature (var={curvature_var:.3f})")

            # Normalized angular density check (not raw count)
            if normalized_angular > 2.0 and symmetry > 0.5:
                issues.append(f"ANGULAR_SYMMETRIC: Angular shape with symmetry (norm_angular={normalized_angular:.2f})")

            # Check for blob (no structure)
            if solidity > self.solidity_blob:
                issues.append(f"BLOB_ARTIFACT: Amorphous mass lacking structure (solidity={solidity:.2f})")
                return ProminenceType.BLOB, 0.75, issues

            # Lack of internal structure is the key indicator
            issues.append(f"UNIFORM_TEXTURE: Lacks filamentary structure (var={internal_var:.3f}, expected >2.0)")

        # Moderate texture - ambiguous case
        else:
            # Check for concerning patterns even with moderate texture
            if symmetry > 0.8:  # Very high symmetry is always suspicious
                issues.append(f"HIGH_SYMMETRY: Unusual bilateral symmetry ({symmetry:.2f})")

        # Classify based on accumulated issues
        if len(issues) == 0:
            # Determine natural type based on shape
            if aspect_ratio > 3:
                return ProminenceType.NATURAL_FILAMENT, 0.85, []
            elif region["curvature"]["mean"] > 0.2:
                return ProminenceType.NATURAL_LOOP, 0.85, []
            else:
                return ProminenceType.NATURAL_ERUPTION, 0.80, []

        # Multiple issues with synthetic texture = likely artifact
        if has_synthetic_texture and len(issues) >= 2:
            return ProminenceType.ANGULAR, 0.75, issues

        # Some issues but not definitive
        confidence = min(0.7, len(issues) * 0.2)
        return ProminenceType.UNKNOWN, confidence, issues

    def analyze_prominences(self, image: np.ndarray) -> Dict:
        """
        Full prominence analysis pipeline.

        Args:
            image: RGB image of sun/render

        Returns:
            Analysis results with detected prominences and issues
        """
        # Detect solar disk
        disk_center, disk_radius = self.detect_solar_disk(image)

        if disk_center is None:
            return {
                "status": "error",
                "message": "Could not detect solar disk",
                "prominences": [],
                "overall_score": 0,
                "issues": ["DISK_NOT_FOUND"]
            }

        # Extract prominence mask
        prominence_mask = self.extract_prominence_mask(image, disk_center, disk_radius)

        # Find individual prominences
        contours, _ = cv2.findContours(prominence_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter small contours
        min_area = (disk_radius * 0.02) ** 2  # At least 2% of disk radius squared
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not contours:
            return {
                "status": "no_prominences",
                "message": "No prominent features detected beyond limb",
                "disk_center": disk_center,
                "disk_radius": disk_radius,
                "prominences": [],
                "overall_score": 50,  # Neutral - no prominences to evaluate
                "issues": []
            }

        # Analyze each prominence
        prominence_results = []
        all_issues = []

        for contour in contours:
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Basic shape metrics
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)

            # Centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w/2, y + h/2

            # Angle from disk center
            dx = cx - disk_center[0]
            dy = cy - disk_center[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi

            # Distance from limb
            dist_from_center = np.sqrt(dx**2 + dy**2)
            dist_from_limb = dist_from_center - disk_radius

            # Aspect ratio
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

            # Curvature analysis
            curvature = self.analyze_contour_curvature(contour)

            # Symmetry
            symmetry = self.compute_symmetry(contour, (x, y, w, h))

            # Cat ear detection
            is_cat_ear, cat_ear_conf = self.detect_cat_ear_pattern(contour, (x, y, w, h))

            # Internal texture
            texture = self.analyze_internal_texture(image, contour, (x, y, w, h))

            # Build region dict for classification
            region_data = {
                "symmetry": symmetry,
                "curvature": curvature,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "texture": texture,
                "area": area,  # Pass area for normalized angular calculation
                "is_cat_ear": is_cat_ear,
                "cat_ear_confidence": cat_ear_conf
            }

            # Classify
            prom_type, confidence, issues = self.classify_prominence(region_data)
            all_issues.extend(issues)

            prominence_results.append({
                "bbox": [x, y, w, h],
                "area": float(area),
                "centroid": [float(cx), float(cy)],
                "angle_from_center": float(angle),
                "distance_from_limb": float(dist_from_limb),
                "aspect_ratio": float(aspect_ratio),
                "solidity": float(solidity),
                "symmetry": float(symmetry),
                "curvature": curvature,
                "texture": texture,
                "is_cat_ear": is_cat_ear,
                "cat_ear_confidence": float(cat_ear_conf),
                "classification": prom_type.value,
                "confidence": float(confidence),
                "issues": issues
            })

        # Compute overall score
        # Start at 100, deduct for issues
        score = 100.0

        # Count artifact types
        artifact_count = sum(1 for p in prominence_results
                            if p["classification"] in ["cat_ear", "ribbon", "blob", "angular"])
        natural_count = sum(1 for p in prominence_results
                           if p["classification"].startswith("natural"))

        # Deductions
        score -= artifact_count * 25  # Major penalty for artifacts
        score -= len(all_issues) * 5  # Smaller penalty for other issues

        # Bonus for natural prominences
        score += natural_count * 5

        score = max(0, min(100, score))

        return {
            "status": "analyzed",
            "disk_center": list(disk_center),
            "disk_radius": int(disk_radius),
            "prominence_count": len(prominence_results),
            "artifact_count": artifact_count,
            "natural_count": natural_count,
            "prominences": prominence_results,
            "overall_score": float(score),
            "issues": all_issues,
            "summary": self._generate_summary(prominence_results, all_issues, score)
        }

    def _generate_summary(self, prominences: List[Dict], issues: List[str], score: float) -> str:
        """Generate human-readable summary."""
        if not prominences:
            return "No prominences detected for analysis."

        lines = [f"Analyzed {len(prominences)} prominence regions."]

        # Count by type
        type_counts = {}
        for p in prominences:
            t = p["classification"]
            type_counts[t] = type_counts.get(t, 0) + 1

        for ptype, count in type_counts.items():
            lines.append(f"  - {ptype}: {count}")

        if issues:
            lines.append(f"\nDetected {len(issues)} morphological issues:")
            for issue in issues[:5]:  # Limit to top 5
                lines.append(f"  - {issue}")
            if len(issues) > 5:
                lines.append(f"  ... and {len(issues) - 5} more")
        else:
            lines.append("\nNo morphological anomalies detected.")

        lines.append(f"\nProminence Shape Score: {score:.1f}/100")

        if score >= 80:
            lines.append("Assessment: Prominences appear natural and realistic.")
        elif score >= 60:
            lines.append("Assessment: Minor shape anomalies detected.")
        elif score >= 40:
            lines.append("Assessment: Significant morphological issues present.")
        else:
            lines.append("Assessment: Severe synthetic artifacts in prominence shapes.")

        return "\n".join(lines)

    def visualize_analysis(self, image: np.ndarray, analysis: Dict,
                            output_path: Optional[str] = None) -> np.ndarray:
        """
        Create visualization of prominence analysis.

        Color coding:
        - Green: Natural prominences
        - Yellow: Minor issues
        - Red: Artifacts (cat ear, ribbon, etc.)
        """
        vis = image.copy()

        if analysis["status"] != "analyzed":
            return vis

        # Draw disk outline
        center = tuple(analysis["disk_center"])
        radius = analysis["disk_radius"]
        cv2.circle(vis, center, radius, (100, 100, 100), 2)

        # Draw each prominence
        for prom in analysis["prominences"]:
            x, y, w, h = prom["bbox"]
            classification = prom["classification"]

            # Color based on classification
            if classification.startswith("natural"):
                color = (0, 255, 0)  # Green
                label = "Natural"
            elif classification == "cat_ear":
                color = (0, 0, 255)  # Red
                label = "CAT EAR!"
            elif classification in ["ribbon", "blob", "angular"]:
                color = (0, 128, 255)  # Orange
                label = classification.upper()
            else:
                color = (0, 255, 255)  # Yellow
                label = "Unknown"

            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)

            # Add label
            cv2.putText(vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1, cv2.LINE_AA)

            # Add confidence
            conf_text = f"{prom['confidence']:.0%}"
            cv2.putText(vis, conf_text, (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1, cv2.LINE_AA)

        # Add overall score
        score = analysis["overall_score"]
        score_color = (0, 255, 0) if score >= 70 else ((0, 255, 255) if score >= 40 else (0, 0, 255))
        cv2.putText(vis, f"Shape Score: {score:.0f}/100", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2, cv2.LINE_AA)

        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        return vis


def analyze_image(image_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Convenience function for single image analysis.

    Args:
        image_path: Path to solar image
        output_path: Optional path to save visualization

    Returns:
        Analysis results dictionary
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"status": "error", "message": f"Could not load image: {image_path}"}

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    classifier = ProminenceShapeClassifier()
    analysis = classifier.analyze_prominences(image)

    if output_path:
        classifier.visualize_analysis(image, analysis, output_path)

    return analysis


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prominence_shape_classifier.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_image(image_path, output_path)
    print(json.dumps(result, indent=2))
