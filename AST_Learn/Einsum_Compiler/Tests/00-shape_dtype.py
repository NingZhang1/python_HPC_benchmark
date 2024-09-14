import unittest

# Import the required classes and functions from the module
# Replace 'your_module' with the actual name of the module where your code resides
from Einsum_Compiler.Shape_Dtype import (
    DtypeProxy,
    F32,
    F64,
    Z64,
    Z128,
    promote_types,
)


class TestPromoteTypes(unittest.TestCase):
    def test_promote_same_kind_higher_precision(self):
        # Test promoting real types with higher precision
        result = promote_types(F32, F64)
        self.assertEqual(result, F64)
        print(f"Promoting F32 and F64 yields: {result.__name__}")  # Expected: F64

    def test_promote_different_kind(self):
        # Test promoting real and complex types
        result = promote_types(F32, Z64)
        self.assertEqual(result, Z64)
        print(f"Promoting F32 and Z64 yields: {result.__name__}")  # Expected: Z64

    def test_promote_complex_types(self):
        # Test promoting complex types with different precisions
        result = promote_types(Z64, Z128)
        self.assertEqual(result, Z128)
        print(f"Promoting Z64 and Z128 yields: {result.__name__}")  # Expected: Z128

    def test_promote_real_and_complex(self):
        # Test promoting real and complex types with higher precision
        result = promote_types(F64, Z128)
        self.assertEqual(result, Z128)
        print(f"Promoting F64 and Z128 yields: {result.__name__}")  # Expected: Z128

    def test_promote_same_type(self):
        # Test promoting the same types
        result = promote_types(F32, F32)
        self.assertEqual(result, F32)
        print(f"Promoting F32 and F32 yields: {result.__name__}")  # Expected: F32

    def test_promote_invalid_type(self):
        # Test promoting with an invalid type
        with self.assertRaises(TypeError):
            promote_types(F32, None)  # NoneType does not have 'kind' or 'precision'

    def test_promote_non_dtype_proxy(self):
        # Test promoting with an object that is not a DtypeProxy subclass
        class NotADtype:
            kind = "real"
            precision = 64

        with self.assertRaises(TypeError):
            promote_types(F32, NotADtype)

    def test_promote_missing_kind(self):
        # Test promoting with a class missing 'kind' attribute
        class MissingKind(DtypeProxy):
            precision = 64

        with self.assertRaises(TypeError):
            promote_types(F32, MissingKind)

    def test_promote_missing_precision(self):
        # Test promoting with a class missing 'precision' attribute
        class MissingPrecision(DtypeProxy):
            kind = "real"

        with self.assertRaises(TypeError):
            promote_types(F32, MissingPrecision)

    def test_promote_invalid_precision_value(self):
        # Test promoting with an invalid 'precision' value
        class InvalidPrecision(DtypeProxy):
            kind = "real"
            precision = "high"

        with self.assertRaises(TypeError):
            promote_types(F32, InvalidPrecision)


if __name__ == "__main__":
    unittest.main()
