import random

# from collections import defaultdict


# Shape proxies
class ShapeProxy:
    pass


class NOCC(ShapeProxy):
    pass


class NVIR(ShapeProxy):
    pass


class NMO(ShapeProxy):
    pass


class NAO(ShapeProxy):
    pass


class N2C(ShapeProxy):
    pass


class N2O(ShapeProxy):
    pass


# Dtype proxies
class DtypeProxy:
    kind = None
    precision = None


class F32(DtypeProxy):
    kind = "real"
    precision = 32


class F64(DtypeProxy):
    kind = "real"
    precision = 64


class Z64(DtypeProxy):
    kind = "complex"
    precision = 64


class Z128(DtypeProxy):
    kind = "complex"
    precision = 128


def promote_types(t1, t2):
    """
    Determine the resulting dtype when combining two dtypes.
    """
    
    if t1 not in [F32, F64, Z64, Z128] or t2 not in [F32, F64, Z64, Z128]:
        raise TypeError("Invalid dtype")
    
    kind1 = t1.kind
    kind2 = t2.kind
    total_prec1 = t1.precision
    total_prec2 = t2.precision
    comp_prec1 = total_prec1 if kind1 == "real" else total_prec1 // 2
    comp_prec2 = total_prec2 if kind2 == "real" else total_prec2 // 2

    if kind1 == kind2:
        return t1 if comp_prec1 >= comp_prec2 else t2
    else:
        max_comp_prec = max(comp_prec1, comp_prec2)
        if max_comp_prec <= 32:
            return Z64
        else:
            return Z128


if __name__ == "__main__":
    # Simple tests for the promote_types function

    # Test 1: Promote F32 and F64
    result_dtype1 = promote_types(F32, F64)
    print(f"Result of promoting F32 and F64: {result_dtype1.__name__}")  # Expected: F64

    # Test 2: Promote F32 and Z64
    result_dtype2 = promote_types(F32, Z64)
    print(f"Result of promoting F32 and Z64: {result_dtype2.__name__}")  # Expected: Z64

    # Test 3: Promote Z64 and Z128
    result_dtype3 = promote_types(Z64, Z128)
    print(
        f"Result of promoting Z64 and Z128: {result_dtype3.__name__}"
    )  # Expected: Z128

    # Test 4: Promote F64 and Z128
    result_dtype4 = promote_types(F64, Z128)
    print(
        f"Result of promoting F64 and Z128: {result_dtype4.__name__}"
    )  # Expected: Z128

    # Test 5: Promote F32 and F32
    result_dtype5 = promote_types(F32, F32)
    print(f"Result of promoting F32 and F32: {result_dtype5.__name__}")  # Expected: F32

    # Note: Since ScalarProxy and TensorProxy are not defined here, we cannot perform tests involving them.
