import unittest
from collections import defaultdict
import re

# Import the required classes and functions
# Replace 'your_module' with the actual name of the module where your code resides
from Einsum_Compiler.Shape_Dtype import promote_types, NOCC, NVIR, NMO, F32, F64, DtypeProxy
from Einsum_Compiler.Scalar_Tensor import TensorProxy, ScalarProxy, generate_name
from Einsum_Compiler.Einsum import einsum

# Test suite
class TestEinsumFunction(unittest.TestCase):
    def setUp(self):
        self.NOCC = NOCC
        self.NVIR = NVIR
        self.NMO = NMO

        self.F32 = F32
        self.F64 = F64

        # Create sample tensors
        self.tensor_a = TensorProxy(name='A', shape=[self.NOCC, self.NVIR], dtype=self.F32)
        self.tensor_b = TensorProxy(name='B', shape=[self.NVIR, self.NMO], dtype=self.F64)
        self.tensor_c = TensorProxy(name='C', shape=[self.NOCC, self.NVIR], dtype=self.F64)
        self.tensor_d = TensorProxy(name='D', shape=[self.NMO, self.NMO], dtype=self.F32)

    def test_correct_einsum(self):
        # Test a correct einsum operation
        result = einsum('ij,jk->ik', self.tensor_a, self.tensor_b)
        self.assertIsInstance(result, TensorProxy)
        self.assertEqual(result.shape, [self.NOCC, self.NMO])
        self.assertEqual(result.dtype, self.F64)
        print(f"Correct einsum result: {result}")

    def test_invalid_operand_type(self):
        # Test with an invalid operand type
        with self.assertRaises(TypeError):
            einsum('ij,jk->ik', self.tensor_a, "not a tensor")

    def test_mismatched_subscripts_operands(self):
        # Test with mismatched number of subscripts and operands
        with self.assertRaises(ValueError):
            einsum('ij,jk->ik', self.tensor_a)

    def test_inconsistent_dimensions(self):
        # Test with inconsistent dimensions for a label
        tensor_e = TensorProxy(name='E', shape=[self.NOCC, self.NOCC], dtype=self.F32)
        with self.assertRaises(ValueError):
            einsum('ij,ij->ij', self.tensor_a, tensor_e)  # NVIR vs NOCC mismatch

    def test_invalid_subscript_length(self):
        # Test with subscript length not matching tensor dimensions
        with self.assertRaises(ValueError):
            einsum('ijk,jk->ik', self.tensor_a, self.tensor_b)  # tensor_a is 2D, subscript is 3D

    def test_invalid_subscript_characters(self):
        # Test with invalid characters in subscripts
        with self.assertRaises(ValueError):
            einsum('i$,jk->ik', self.tensor_a, self.tensor_b)  # Invalid character '$'

    def test_missing_output_subscript(self):
        # Test with missing output subscript
        result = einsum('ij,jk', self.tensor_a, self.tensor_b)
        self.assertIsInstance(result, TensorProxy)
        print(f"Einsum without explicit output subscript result: {result}")

    def test_duplicate_output_labels(self):
        # Test with duplicate labels in the output
        with self.assertRaises(ValueError):
            einsum('ij,jk->iik', self.tensor_a, self.tensor_b)  # Duplicate 'i' in output

    def test_invalid_operand_dimensions(self):
        # Test with tensors having mismatched dimensions
        tensor_f = TensorProxy(name='F', shape=[self.NOCC], dtype=self.F32)
        with self.assertRaises(ValueError):
            einsum('ij,jk->ik', tensor_f, self.tensor_b)  # tensor_f has insufficient dimensions

    def test_missing_labels_in_operands(self):
        # Test with labels in subscripts not present in operands
        with self.assertRaises(ValueError):
            einsum('ij,jk->il', self.tensor_a, self.tensor_b)  # 'l' not defined in input subscripts

    def test_empty_subscripts(self):
        # Test with empty subscripts
        with self.assertRaises(ValueError):
            einsum('', self.tensor_a)

    def test_non_string_subscripts(self):
        # Test with non-string subscripts
        with self.assertRaises(AttributeError):
            einsum(None, self.tensor_a)

    def test_scalar_in_einsum(self):
        # Test using a scalar in einsum (should raise TypeError)
        scalar = ScalarProxy(dtype=self.F32, name='scalar')
        with self.assertRaises(TypeError):
            einsum('i,i->', self.tensor_a, scalar)

    def test_invalid_label_dimensions(self):
        # Test with labels mapping to different dimensions
        tensor_g = TensorProxy(name='G', shape=[self.NMO, self.NVIR], dtype=self.F32)
        with self.assertRaises(ValueError):
            einsum('ij,ij->ij', self.tensor_a, tensor_g)  # NOCC vs NMO mismatch for 'i'

    def test_output_label_not_in_inputs(self):
        # Test with output label not present in inputs
        with self.assertRaises(ValueError):
            einsum('ij,jk->il', self.tensor_a, self.tensor_b)  # 'l' not present in inputs

    def test_invalid_dtype_promotion(self):
        # Test promoting invalid dtypes
        class InvalidDtype(DtypeProxy):
            kind = 'invalid'
            precision = 64

        tensor_invalid = TensorProxy(name='Invalid', shape=[self.NOCC, self.NVIR], dtype=InvalidDtype)
        with self.assertRaises(TypeError):
            einsum('ij,ij->ij', self.tensor_a, tensor_invalid)

# Run the tests
if __name__ == '__main__':
    unittest.main()
