from Einsum_Compiler.Shape_Dtype import ShapeProxy, DtypeProxy, promote_types

# Global variable for controlling the naming of new tensors and scalars
NAME_COUNTER = 0


def generate_name(prefix="%"):
    global NAME_COUNTER
    NAME_COUNTER += 1
    return f"{prefix}{NAME_COUNTER}"


# Scalar proxy
class ScalarProxy:
    def __init__(self, name=None, dtype=None):
        assert issubclass(dtype, DtypeProxy)
        self.dtype = dtype
        self.name = name if name else generate_name()

    # Addition
    def __add__(self, other):
        if isinstance(other, ScalarProxy):
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return ScalarProxy(dtype=result_dtype, name=result_name)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ScalarProxy' and '{type(other).__name__}'"
            )

    def __iadd__(self, other):
        if isinstance(other, ScalarProxy):
            self.dtype = promote_types(self.dtype, other.dtype)
            return self
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +=: 'ScalarProxy' and '{type(other).__name__}'"
            )

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, ScalarProxy):
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return ScalarProxy(dtype=result_dtype, name=result_name)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ScalarProxy' and '{type(other).__name__}'"
            )

    def __isub__(self, other):
        if isinstance(other, ScalarProxy):
            self.dtype = promote_types(self.dtype, other.dtype)
            return self
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -=: 'ScalarProxy' and '{type(other).__name__}'"
            )

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, ScalarProxy):
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return ScalarProxy(dtype=result_dtype, name=result_name)
        elif isinstance(other, TensorProxy):
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return TensorProxy(name=result_name, shape=other.shape, dtype=result_dtype)
        else:
            raise TypeError(f"Cannot multiply ScalarProxy with {type(other).__name__}")

    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, ScalarProxy):
            self.dtype = promote_types(self.dtype, other.dtype)
            return self
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *=: 'ScalarProxy' and '{type(other).__name__}'"
            )

    # Division
    def __truediv__(self, other):
        if isinstance(other, ScalarProxy):
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return ScalarProxy(dtype=result_dtype, name=result_name)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for /: 'ScalarProxy' and '{type(other).__name__}'"
            )

    def __itruediv__(self, other):
        if isinstance(other, ScalarProxy):
            self.dtype = promote_types(self.dtype, other.dtype)
            return self
        else:
            raise TypeError(
                f"Unsupported operand type(s) for /=: 'ScalarProxy' and '{type(other).__name__}'"
            )

    def __repr__(self):
        return f"ScalarProxy(name={self.name}, dtype={self.dtype.__name__})"


# Tensor proxy
class TensorProxy:
    def __init__(self, name, shape, dtype):
        self.name = name
        if not isinstance(shape, list):
            shape = [shape]
        else:
            assert len(shape) > 0
        assert all(isinstance(i, type) for i in shape)
        assert all(
            [issubclass(s, ShapeProxy) for s in shape]
        ), "Shape elements must be ShapeProxy subclasses."
        self.shape = shape
        assert isinstance(dtype, type)
        assert issubclass(dtype, DtypeProxy), "dtype must be a DtypeProxy subclass."
        self.dtype = dtype

    # Addition
    def __add__(self, other):
        if isinstance(other, TensorProxy):
            if self.shape != other.shape:
                raise ValueError("Shapes must be the same for element-wise addition.")
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return TensorProxy(name=result_name, shape=self.shape, dtype=result_dtype)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: 'TensorProxy' and '{type(other).__name__}'"
            )

    def __iadd__(self, other):
        if isinstance(other, TensorProxy):
            if self.shape != other.shape:
                raise ValueError("Shapes must be the same for element-wise addition.")
            self.dtype = promote_types(self.dtype, other.dtype)
            return self
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +=: 'TensorProxy' and '{type(other).__name__}'"
            )

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, TensorProxy):
            if self.shape != other.shape:
                raise ValueError(
                    "Shapes must be the same for element-wise subtraction."
                )
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return TensorProxy(name=result_name, shape=self.shape, dtype=result_dtype)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -: 'TensorProxy' and '{type(other).__name__}'"
            )

    def __isub__(self, other):
        if isinstance(other, TensorProxy):
            if self.shape != other.shape:
                raise ValueError(
                    "Shapes must be the same for element-wise subtraction."
                )
            self.dtype = promote_types(self.dtype, other.dtype)
            return self
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -=: 'TensorProxy' and '{type(other).__name__}'"
            )

    # Multiplication (element-wise and scalar multiplication)
    def __mul__(self, other):
        if isinstance(other, ScalarProxy):
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return TensorProxy(name=result_name, shape=self.shape, dtype=result_dtype)
        elif isinstance(other, TensorProxy):
            if self.shape != other.shape:
                raise ValueError(
                    "Shapes must be the same for element-wise multiplication."
                )
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return TensorProxy(name=result_name, shape=self.shape, dtype=result_dtype)
        else:
            raise TypeError(f"Cannot multiply TensorProxy with {type(other).__name__}")

    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, (ScalarProxy, TensorProxy)):
            if isinstance(other, TensorProxy) and self.shape != other.shape:
                raise ValueError(
                    "Shapes must be the same for element-wise multiplication."
                )
            self.dtype = promote_types(self.dtype, other.dtype)
            return self
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *=: 'TensorProxy' and '{type(other).__name__}'"
            )

    # Division
    def __truediv__(self, other):
        if isinstance(other, (ScalarProxy, TensorProxy)):
            if isinstance(other, TensorProxy) and self.shape != other.shape:
                raise ValueError("Shapes must be the same for element-wise division.")
            result_dtype = promote_types(self.dtype, other.dtype)
            result_name = generate_name()
            return TensorProxy(name=result_name, shape=self.shape, dtype=result_dtype)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for /: 'TensorProxy' and '{type(other).__name__}'"
            )

    def __itruediv__(self, other):
        if isinstance(other, (ScalarProxy, TensorProxy)):
            if isinstance(other, TensorProxy) and self.shape != other.shape:
                raise ValueError("Shapes must be the same for element-wise division.")
            self.dtype = promote_types(self.dtype, other.dtype)
            return self
        else:
            raise TypeError(
                f"Unsupported operand type(s) for /=: 'TensorProxy' and '{type(other).__name__}'"
            )

    # Dot product
    def dot(self, other, axes=None):
        if not isinstance(other, TensorProxy):
            raise TypeError(f"Cannot perform dot product with {type(other).__name__}")

        # Default axes: last axis of self, first axis of other
        if axes is None:
            axes_self = [len(self.shape) - 1]
            axes_other = [0]
        elif isinstance(axes, int):
            axes_self = list(range(len(self.shape) - axes, len(self.shape)))
            axes_other = list(range(axes))
        elif isinstance(axes, (list, tuple)) and len(axes) == 2:
            axes_self, axes_other = axes
            if isinstance(axes_self, int):
                axes_self = [axes_self]
            if isinstance(axes_other, int):
                axes_other = [axes_other]
        else:
            raise ValueError("Invalid axes specification.")

        if len(axes_self) != len(axes_other):
            raise ValueError("Axes lengths mismatch.")

        # Check shape compatibility
        for i, j in zip(axes_self, axes_other):
            if self.shape[i] != other.shape[j]:
                raise ValueError(
                    f"Shape mismatch on axes {i} and {j}: "
                    f"{self.shape[i].__name__} != {other.shape[j].__name__}"
                )

        # Compute the resulting shape
        new_shape_self = [s for idx, s in enumerate(self.shape) if idx not in axes_self]
        new_shape_other = [
            s for idx, s in enumerate(other.shape) if idx not in axes_other
        ]
        result_shape = new_shape_self + new_shape_other

        # Promote dtype
        result_dtype = promote_types(self.dtype, other.dtype)

        # Generate a new name for the result tensor
        result_name = generate_name()

        # Return new TensorProxy
        return TensorProxy(name=result_name, shape=result_shape, dtype=result_dtype)

    def __repr__(self):
        shape_str = ", ".join([s.__name__ for s in self.shape])
        return f"TensorProxy(name={self.name}, shape=[{shape_str}], dtype={self.dtype.__name__})"


if __name__ == "__main__":

    from Einsum_Compiler.Shape_Dtype import NOCC, NVIR, NMO, F32, F64, promote_types

    # Test ScalarProxy
    scalar_a = ScalarProxy(dtype=F32, name="alpha")
    scalar_b = ScalarProxy(dtype=F64, name="beta")

    # Test addition
    scalar_c = scalar_a + scalar_b
    print(f"Scalar addition result: {scalar_c}")

    # Test multiplication
    scalar_d = scalar_a * scalar_b
    print(f"Scalar multiplication result: {scalar_d}")

    # Test TensorProxy
    tensor_a = TensorProxy(name="A", shape=[NOCC, NVIR], dtype=F32)
    tensor_b = TensorProxy(name="B", shape=[NOCC, NVIR], dtype=F64)

    # Test tensor addition
    tensor_c = tensor_a + tensor_b
    print(f"Tensor addition result: {tensor_c}")

    # Test tensor and scalar multiplication
    tensor_d = tensor_a * scalar_b
    print(f"Tensor and scalar multiplication result: {tensor_d}")

    # Test dot product
    tensor_e = TensorProxy(name="E", shape=[NVIR, NMO], dtype=F32)
    tensor_f = tensor_a.dot(tensor_e)
    print(f"Tensor dot product result: {tensor_f}")
