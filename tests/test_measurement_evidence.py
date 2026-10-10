"""Regression tests for measurement provenance in orbit views."""

import unittest

from backend.fusion.robust_fuser import RobustMeasurementFuser, ViewObservation
from backend.video.geometry import MaskGeometry, OrbitViewGeometry
from backend.video.measurement import INSUFFICIENT, _measure_view


class TestTempleMeasurementEvidence(unittest.TestCase):
    def test_short_profile_blob_is_not_reported_as_measured_temple_length(self) -> None:
        geometry = MaskGeometry(
            x=0,
            y=0,
            width=150,
            height=48,
            area=2600,
            image_h=240,
            image_w=320,
            bbox_fill=0.36,
            aspect=3.125,
            coverage=0.034,
            max_column_thickness=22,
            thin_column_ratio=0.2,
            temple_visibility=0.8,
            lens_cluster_width=135,
            lens_cluster_height=42,
            temple_span=80,
        )
        view = OrbitViewGeometry(
            index=4,
            mask_geometry=geometry,
            yaw_deg=90.0,
            cos_yaw=0.0,
        )

        result = _measure_view(
            view,
            geometry,
            label="side",
            confidence=0.9,
            quality=0.9,
            mm_per_px=1.0,
            estimated_rim_band_px=1.0,
        )

        self.assertEqual(result.evidence["temple_length"], INSUFFICIENT)
        self.assertNotIn("temple_length", result.observation_map())

        fused = RobustMeasurementFuser().fuse([
            ViewObservation(
                view="side",
                measurements=result.measurements,
                weight=0.9,
                evidence=result.evidence,
            )
        ])
        report = fused.dimensions["temple_length"]
        self.assertEqual(report.source, "insufficient_evidence")
        self.assertIn("estimated from frame width", report.warnings[0])
        self.assertAlmostEqual(report.value, 130.95, places=1)


if __name__ == "__main__":
    unittest.main()
