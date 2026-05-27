import unittest
from pathlib import Path

import numpy as np

from MIRI_cruciform_diffractio import (
    MIRICruciform,
    stellar_blackbody_spectrum,
    lrs_detector_offset_pixels,
    lrs_slitless_dispersion_angle,
    um,
)


class TestLRSSlitlessHelpers(unittest.TestCase):
    def test_default_data_paths_use_repository_data(self):
        model = MIRICruciform(mode="IMA", simsize=1024)

        self.assertEqual(model.jwst_pupil.shape, (1024, 1024))
        self.assertTrue(Path(model.filterfiles["F560W"]).is_file())
        self.assertTrue(Path(model.filterfiles["F770W"]).is_file())

    def test_lrs_dispersion_is_zero_at_reference_wavelength(self):
        self.assertAlmostEqual(lrs_detector_offset_pixels(8.4), 0.0, places=12)
        self.assertAlmostEqual(lrs_slitless_dispersion_angle(8.4, focal_length_um=7e6), 0.0, places=12)

        model = MIRICruciform(mode="LRS-SLTSS", simsize=1024)
        model.dispersion(8.4 * um, mode="LRS-SLTSS")
        self.assertAlmostEqual(model.dispersion_angle, 0.0, places=12)

    def test_lrs_dispersion_angle_grows_away_from_reference(self):
        angle_near = lrs_slitless_dispersion_angle(9.0, focal_length_um=7e6)
        angle_far = lrs_slitless_dispersion_angle(12.0, focal_length_um=7e6)

        self.assertLess(angle_near, 0.0)
        self.assertLess(angle_far, 0.0)
        self.assertGreater(abs(angle_far), abs(angle_near))

    def test_stellar_blackbody_spectrum_is_normalized(self):
        spectrum = stellar_blackbody_spectrum(np.array([5.0, 8.0, 12.0]), temperature=5000.0)

        self.assertTrue(np.all(spectrum > 0.0))
        self.assertAlmostEqual(float(spectrum.max()), 1.0, places=12)

    def test_lrssim_applies_spectral_weights(self):
        model = object.__new__(MIRICruciform)

        def fake_monochromatic_cruciform(self, wavelength, detector_angle=0.0, tr_radius=0.0):
            base = np.full((2, 2), wavelength)
            return base, base * 10, base * 100, base * 1000

        model.monochromatic_cruciform = fake_monochromatic_cruciform.__get__(model, MIRICruciform)

        components = MIRICruciform.LRSsim(
            model,
            wavelengths=np.array([5.0, 6.0]),
            weights=np.array([1.0, 0.5]),
            detector_angle=1.5,
            tr_radius=10.0,
        )

        np.testing.assert_allclose(components[0], np.full((2, 2), 8.0))
        np.testing.assert_allclose(components[1], np.full((2, 2), 80.0))
        np.testing.assert_allclose(components[2], np.full((2, 2), 800.0))
        np.testing.assert_allclose(components[3], np.full((2, 2), 8000.0))

    def test_lrssim_rejects_mismatched_weights(self):
        model = object.__new__(MIRICruciform)
        model.monochromatic_cruciform = lambda *args, **kwargs: (0, 0, 0, 0)

        with self.assertRaises(ValueError):
            MIRICruciform.LRSsim(model, wavelengths=np.array([5.0, 6.0]), weights=np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
