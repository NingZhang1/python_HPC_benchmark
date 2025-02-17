import unittest
import numpy as np
import BackEnd._config as CONFIG

CONFIG.enable_fftw()
# CONFIG.disable_fftw()
# CONFIG.backend("torch")
CONFIG.backend("scipy")

import BackEnd.isdf_backend as BackEnd
from BackEnd.isdf_fft_cache import DynamicCached3DRFFT, FFTW_FOUND

ToNUMPY = BackEnd._toNumpy


class TestDynamicCached3DRFFT(unittest.TestCase):
    def setUp(self):
        self.fft_calculator = DynamicCached3DRFFT(initial_shape=(64, 64, 64))

    def test_rfft_memory_usage(self):
        if not FFTW_FOUND:
            print("FFTW not available")
            if not CONFIG.USE_TORCH:
                self.skipTest("FFTW not available")
        else:
            print("FFTW available")

        input_data = np.random.rand(64, 64, 64)
        input_data = BackEnd._toTensor(input_data)
        result = self.fft_calculator.rfft(input_data)

        if isinstance(result, np.ndarray):
            self.assertEqual(
                result.__array_interface__["data"][0],
                self.fft_calculator.complex_buffer.__array_interface__["data"][0],
                "RFFT result should use the complex_buffer",
            )
        else:
            self.assertEqual(
                result.data_ptr(),
                BackEnd._toTensor(self.fft_calculator.complex_buffer).data_ptr(),
                "RFFT result should use the complex_buffer",
            )

    def test_irfft_memory_usage(self):
        if not FFTW_FOUND:
            if not CONFIG.USE_TORCH:
                self.skipTest("FFTW not available")
            # self.skipTest("FFTW not available")

        input_data = np.random.rand(64, 64, 33) + 1j * np.random.rand(64, 64, 33)
        input_data = BackEnd._toTensor(input_data)
        result = self.fft_calculator.irfft(input_data, s=(64, 64, 64))

        if isinstance(result, np.ndarray):
            self.assertEqual(
                result.__array_interface__["data"][0],
                self.fft_calculator.real_buffer.__array_interface__["data"][0],
                "IRFFT result should use the real_buffer",
            )
        else:
            self.assertEqual(
                result.data_ptr(),
                BackEnd._toTensor(self.fft_calculator.real_buffer).data_ptr(),
                "IRFFT result should use the real_buffer",
            )

    def test_rfft_irfft_correctness(self):
        input_data = np.random.rand(64, 64, 64)
        input_data = BackEnd._toTensor(input_data)
        fft_result = self.fft_calculator.rfft(input_data)
        reconstructed = self.fft_calculator.irfft(fft_result, s=input_data.shape)
        reconstructed = ToNUMPY(reconstructed)
        np.testing.assert_allclose(input_data, reconstructed, rtol=1e-10, atol=1e-10)

    def test_different_shapes(self):
        shapes = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                input_data = np.random.rand(*shape)
                input_data = BackEnd._toTensor(input_data)
                fft_result = self.fft_calculator.rfft(input_data)
                reconstructed = self.fft_calculator.irfft(fft_result, s=shape)
                reconstructed = ToNUMPY(reconstructed)
                np.testing.assert_allclose(
                    input_data, reconstructed, rtol=1e-10, atol=1e-10
                )

    def test_4d_input(self):
        input_data = np.random.rand(10, 64, 64, 64)
        input_data = BackEnd._toTensor(input_data)
        fft_result = self.fft_calculator.rfft(input_data)
        reconstructed = self.fft_calculator.irfft(fft_result, s=input_data.shape)
        reconstructed = ToNUMPY(reconstructed)
        np.testing.assert_allclose(input_data, reconstructed, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
