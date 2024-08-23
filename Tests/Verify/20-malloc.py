import numpy as np
import torch
import unittest

from BackEnd._torch import malloc as torch_malloc
from BackEnd._numpy import malloc as numpy_malloc


class TestMallocImplementations(unittest.TestCase):

    def test_basic_allocation(self):
        shape = (3, 4)

        torch_arr = torch_malloc(shape, torch.float64)
        numpy_arr = numpy_malloc(shape, np.float64)

        self.assertEqual(torch_arr.shape, shape)
        self.assertEqual(numpy_arr.shape, shape)
        self.assertEqual(torch_arr.dtype, torch.float64)
        self.assertEqual(numpy_arr.dtype, np.float64)

    def test_allocation_with_buffer(self):
        shape = (2, 3)
        dtype = np.float64
        buf_size = 48  # 6 elements * 8 bytes each

        torch_buf = torch.empty(buf_size // 8, dtype=torch.float64)
        numpy_buf = np.empty(buf_size, dtype=np.float64)

        torch_arr = torch_malloc(shape, torch.float64, buf=torch_buf)
        numpy_arr = numpy_malloc(shape, np.float64, buf=numpy_buf)

        self.assertEqual(torch_arr.shape, shape)
        self.assertEqual(numpy_arr.shape, shape)
        self.assertEqual(torch_arr.dtype, torch.float64)
        self.assertEqual(numpy_arr.dtype, np.float64)

        print("torch_buf_address = ", torch_buf.data_ptr())
        print("torch_arr_address = ", torch_arr.data_ptr())
        print("numpy_buf_address = ", numpy_buf.__array_interface__["data"][0])
        print("numpy_arr_address = ", numpy_arr.__array_interface__["data"][0])

    def test_allocation_with_offset(self):
        shape = (2, 2)
        # dtype = np.float64
        buf_size = 48  # 6 elements * 8 bytes each
        offset = 16  # 2 elements * 8 bytes each

        torch_buf = torch.empty(buf_size // 8, dtype=torch.float64)
        numpy_buf = np.empty(buf_size // 8, dtype=np.float64)

        torch_arr = torch_malloc(shape, torch.float64, buf=torch_buf, offset=offset)
        numpy_arr = numpy_malloc(shape, np.float64, buf=numpy_buf, offset=offset)

        self.assertEqual(torch_arr.shape, shape)
        self.assertEqual(numpy_arr.shape, shape)
        self.assertEqual(torch_arr.dtype, torch.float64)
        self.assertEqual(numpy_arr.dtype, np.float64)

        print("torch_buf_address = ", torch_buf.data_ptr())
        print("torch_arr_address = ", torch_arr.data_ptr())
        print("numpy_buf_address = ", numpy_buf.__array_interface__["data"][0])
        print("numpy_arr_address = ", numpy_arr.__array_interface__["data"][0])

    def test_gpu_allocation(self):

        if torch.cuda.is_available():
            shape = (2, 2)

            torch_arr = torch_malloc(shape, torch.float64, gpu=True)

            self.assertEqual(torch_arr.shape, shape)
            self.assertEqual(torch_arr.dtype, torch.float64)
            self.assertTrue(torch_arr.is_cuda)

        with self.assertRaises(AssertionError):
            numpy_malloc(shape, np.float64, gpu=True)

    def test_error_checking(self):
        shape = (2, 2)
        dtype = np.float64
        buf = torch.empty(4, dtype=torch.float64)

        with self.assertRaises(AssertionError):
            torch_malloc(shape, torch.float64, buf=buf, offset=1)  # Misaligned offset

        with self.assertRaises(AssertionError):
            torch_malloc(shape, np.float32, buf=buf)  # Dtype mismatch


if __name__ == "__main__":
    unittest.main()
