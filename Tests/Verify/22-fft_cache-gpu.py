import unittest
import torch
import BackEnd._config as CONFIG

# CONFIG.enable_fftw()
CONFIG.disable_fftw()
CONFIG.backend("torch_gpu")
CONFIG.disable_pyscf_lib()
import BackEnd._numpy

import BackEnd.isdf_backend as BackEnd
from BackEnd.isdf_fft_cache import DynamicCached3DRFFT_GPU

ToNUMPY = BackEnd._toNumpy


class TestDynamicCached3DRFFT(unittest.TestCase):
    def setUp(self):
        self.fft_calculator = DynamicCached3DRFFT_GPU(initial_shape=(64, 64, 64))

    def test_rfft_memory_usage(self):

        input_data = torch.rand(64, 64, 64, device="cuda", dtype=torch.float64)
        result = self.fft_calculator.rfft(input_data)

        self.assertEqual(
            result.data_ptr(),
            self.fft_calculator.complex_buffer.data_ptr(),
            "RFFT result should use the complex_buffer",
        )

    def test_irfft_memory_usage(self):

        input_data = torch.rand(64, 64, 33, dtype=torch.complex128, device="cuda")
        result = self.fft_calculator.irfft(input_data, s=(64, 64, 64))

        self.assertEqual(
            result.data_ptr(),
            self.fft_calculator.real_buffer.data_ptr(),
            "IRFFT result should use the real_buffer",
        )

    def test_rfft_irfft_correctness(self):
        input_data = torch.rand(64, 64, 64, device="cuda", dtype=torch.float64)
        fft_result = self.fft_calculator.rfft(input_data)
        reconstructed = self.fft_calculator.irfft(fft_result, s=input_data.shape)
        reconstructed = ToNUMPY(reconstructed.cpu())
        input_data_np = ToNUMPY(input_data.cpu())
        torch.testing.assert_close(input_data_np, reconstructed, rtol=1e-5, atol=1e-5)

    def test_different_shapes(self):
        shapes = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
        for shape in shapes:
            with self.subTest(shape=shape):
                input_data = torch.rand(*shape, device="cuda", dtype=torch.float64)
                fft_result = self.fft_calculator.rfft(input_data)
                reconstructed = self.fft_calculator.irfft(fft_result, s=shape)
                reconstructed = ToNUMPY(reconstructed.cpu())
                input_data_np = ToNUMPY(input_data.cpu())
                torch.testing.assert_close(
                    input_data_np, reconstructed, rtol=1e-5, atol=1e-5
                )

    def test_4d_input(self):
        input_data = torch.rand(10, 64, 64, 64, device="cuda", dtype=torch.float64)
        fft_result = self.fft_calculator.rfft(input_data)
        reconstructed = self.fft_calculator.irfft(fft_result, s=input_data.shape)
        reconstructed = ToNUMPY(reconstructed.cpu())
        input_data_np = ToNUMPY(input_data.cpu())
        torch.testing.assert_close(input_data_np, reconstructed, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
