import unittest
from Einsum_Compiler.Shape_Dtype import (
    ShapeProxy,
    DtypeProxy,
    NOCC,
    NVIR,
    NMO,
    F32,
    F64,
    promote_types,
)
from Einsum_Compiler.Scalar_Tensor import (
    ScalarProxy,
    TensorProxy,
    generate_name,
)  # Replace 'your_module' with the actual module name


class TestScalarProxy(unittest.TestCase):
    def setUp(self):
        # Reset the name counter before each test
        global NAME_COUNTER
        NAME_COUNTER = 0

        self.scalar_f32 = ScalarProxy(dtype=F32, name="alpha")
        self.scalar_f64 = ScalarProxy(dtype=F64, name="beta")

    def test_scalar_add_invalid_type(self):
        with self.assertRaises(TypeError):
            result = self.scalar_f32 + 5  # Adding an int, should raise TypeError

    def test_scalar_mul_invalid_type(self):
        with self.assertRaises(TypeError):
            result = (
                self.scalar_f32 * "string"
            )  # Multiplying with a string, should raise TypeError

    def test_scalar_div_invalid_type(self):
        with self.assertRaises(TypeError):
            result = self.scalar_f32 / [
                1,
                2,
                3,
            ]  # Dividing by a list, should raise TypeError

    def test_scalar_iadd_invalid_type(self):
        with self.assertRaises(TypeError):
            self.scalar_f32 += None  # Adding None, should raise TypeError

    def test_scalar_isub_invalid_type(self):
        with self.assertRaises(TypeError):
            self.scalar_f32 -= {}  # Subtracting a dict, should raise TypeError

    def test_scalar_imul_invalid_type(self):
        with self.assertRaises(TypeError):
            self.scalar_f32 *= 3.14  # Multiplying with a float, should raise TypeError

    def test_scalar_itruediv_invalid_type(self):
        with self.assertRaises(TypeError):
            self.scalar_f32 /= True  # Dividing by a bool, should raise TypeError


class TestTensorProxy(unittest.TestCase):
    def setUp(self):
        # Reset the name counter before each test
        global NAME_COUNTER
        NAME_COUNTER = 0

        self.tensor_a = TensorProxy(name="A", shape=[NOCC, NVIR], dtype=F32)
        self.tensor_b = TensorProxy(name="B", shape=[NOCC, NVIR], dtype=F64)
        self.scalar_f32 = ScalarProxy(dtype=F32, name="alpha")

    def test_tensor_add_shape_mismatch(self):
        tensor_c = TensorProxy(name="C", shape=[NVIR, NOCC], dtype=F32)
        with self.assertRaises(ValueError):
            result = self.tensor_a + tensor_c  # Shapes do not match

    def test_tensor_sub_shape_mismatch(self):
        tensor_c = TensorProxy(name="C", shape=[NOCC], dtype=F32)
        with self.assertRaises(ValueError):
            result = self.tensor_a - tensor_c  # Shapes do not match

    def test_tensor_mul_invalid_type(self):
        with self.assertRaises(TypeError):
            result = (
                self.tensor_a * "string"
            )  # Multiplying with a string, should raise TypeError

    def test_tensor_div_invalid_type(self):
        with self.assertRaises(TypeError):
            result = self.tensor_a / None  # Dividing by None, should raise TypeError

    def test_tensor_iadd_invalid_type(self):
        with self.assertRaises(TypeError):
            self.tensor_a += 5  # Adding an int, should raise TypeError

    def test_tensor_isub_invalid_type(self):
        with self.assertRaises(TypeError):
            self.tensor_a -= [1, 2, 3]  # Subtracting a list, should raise TypeError

    def test_tensor_imul_invalid_type(self):
        with self.assertRaises(TypeError):
            self.tensor_a *= {}  # Multiplying with a dict, should raise TypeError

    def test_tensor_itruediv_invalid_type(self):
        with self.assertRaises(TypeError):
            self.tensor_a /= 3.14  # Dividing by a float, should raise TypeError

    def test_tensor_mul_shape_mismatch(self):
        tensor_c = TensorProxy(name="C", shape=[NOCC, NMO], dtype=F32)
        with self.assertRaises(ValueError):
            result = self.tensor_a * tensor_c  # Shapes do not match

    def test_tensor_div_shape_mismatch(self):
        tensor_c = TensorProxy(name="C", shape=[NMO, NVIR], dtype=F32)
        with self.assertRaises(ValueError):
            result = self.tensor_a / tensor_c  # Shapes do not match

    def test_tensor_dot_invalid_type(self):
        with self.assertRaises(TypeError):
            result = self.tensor_a.dot(
                "string"
            )  # Dot product with a string, should raise TypeError

    def test_tensor_dot_axes_length_mismatch(self):
        tensor_c = TensorProxy(name="C", shape=[NVIR, NMO], dtype=F32)
        with self.assertRaises(ValueError):
            result = self.tensor_a.dot(
                tensor_c, axes=([0, 1], [0])
            )  # Axes lengths mismatch

    def test_tensor_dot_shape_mismatch(self):
        tensor_c = TensorProxy(name="C", shape=[NMO, NMO], dtype=F32)
        with self.assertRaises(ValueError):
            result = self.tensor_a.dot(tensor_c)  # Shapes do not align for dot product


class TestInitializationErrors(unittest.TestCase):
    def test_scalar_invalid_dtype(self):
        with self.assertRaises(AssertionError):
            scalar = ScalarProxy(dtype=int)  # int is not a DtypeProxy subclass

    def test_tensor_invalid_dtype(self):
        with self.assertRaises(AssertionError):
            tensor = TensorProxy(
                name="A", shape=[NOCC, NVIR], dtype=float
            )  # float is not a DtypeProxy subclass

    def test_tensor_invalid_shape(self):
        with self.assertRaises(AssertionError):
            tensor = TensorProxy(
                name="A", shape=[NOCC, 5], dtype=F32
            )  # 5 is not a ShapeProxy subclass

    def test_tensor_shape_not_list(self):
        tensor = TensorProxy(name="A", shape=NOCC, dtype=F32)  # shape should be a list

    def test_tensor_shape_empty(self):
        with self.assertRaises(AssertionError):
            tensor = TensorProxy(name="A", shape=[], dtype=F32)  # Empty shape list


class TestPromoteTypesErrors(unittest.TestCase):
    def test_promote_types_invalid_dtype(self):
        with self.assertRaises(AttributeError):
            result = promote_types(
                F32, None
            )  # NoneType does not have required attributes

    def test_promote_types_invalid_precision(self):
        class InvalidDtype(DtypeProxy):
            kind = "real"
            precision = "high"  # Invalid precision

        with self.assertRaises(TypeError):
            result = promote_types(F32, InvalidDtype)


if __name__ == "__main__":
    unittest.main()
