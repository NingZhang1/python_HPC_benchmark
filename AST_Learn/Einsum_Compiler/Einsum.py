from collections import defaultdict  # Needed for defaultdict
import re  # Needed for regular expressions

from Einsum_Compiler.Shape_Dtype import promote_types
from Einsum_Compiler.Scalar_Tensor import TensorProxy, generate_name, ScalarProxy


# Global einsum function
def einsum(subscripts, *operands):
    """
    Perform Einstein summation using the given subscripts and operands.
    """
    # Remove whitespaces
    subscripts = subscripts.replace(" ", "")

    # Split the subscripts into input and output
    if "->" in subscripts:
        input_subs, output_subs = subscripts.split("->")
    else:
        input_subs = subscripts
        output_subs = None

    input_subs_list = input_subs.split(",")

    if len(input_subs_list) != len(operands):
        raise ValueError("Number of operands does not match number of subscripts.")

    # Define a pattern for valid labels (letters only)
    valid_label_pattern = re.compile(r"^[A-Za-z]+$")

    # Build a mapping from subscript labels to dimensions
    label_dim_map = defaultdict(set)
    label_dim_order = {}

    # Promote dtype
    result_dtype = operands[0].dtype
    for operand in operands:
        if not isinstance(operand, TensorProxy):
            raise TypeError("All operands must be TensorProxy instances.")
        result_dtype = promote_types(result_dtype, operand.dtype)

    # Process each operand
    for idx, (subs, tensor) in enumerate(zip(input_subs_list, operands)):
        if len(subs) != len(tensor.shape):
            raise ValueError(
                f"Subscript length does not match tensor dimensions for operand {idx}."
            )

        # Validate the subscripts for each operand
        if not valid_label_pattern.match(subs):
            raise ValueError(f"Invalid labels in subscripts: '{subs}'")

        for label, dim in zip(subs, tensor.shape):
            if not label.isalpha():
                raise ValueError(f"Invalid label '{label}' in subscripts.")
            label_dim_map[label].add(dim)
            if label not in label_dim_order:
                label_dim_order[label] = dim

    # Check that dimensions with the same labels are compatible
    for label, dims in label_dim_map.items():
        if len(dims) > 1:
            raise ValueError(
                f"Inconsistent dimensions for label '{label}': {[d.__name__ for d in dims]}"
            )

    # If output subscripts are not specified, compute the set of labels that appear only once
    if output_subs is None:
        all_labels = "".join(input_subs_list)
        label_counts = defaultdict(int)
        for label in all_labels:
            label_counts[label] += 1
        output_labels = "".join(
            [label for label in all_labels if label_counts[label] == 1]
        )
    else:
        output_labels = output_subs

    # Validate the output labels
    if not valid_label_pattern.match(output_labels):
        raise ValueError(f"Invalid labels in output subscripts: '{output_labels}'")

    for label in output_labels:
        if not label.isalpha():
            raise ValueError(f"Invalid label '{label}' in output subscripts.")
        if label not in label_dim_order:
            raise ValueError(
                f"Output label '{label}' does not appear in input subscripts."
            )

    # Check for duplicate labels in the output subscripts
    if len(set(output_labels)) != len(output_labels):
        duplicates = [
            label for label in set(output_labels) if output_labels.count(label) > 1
        ]
        raise ValueError(f"Duplicate labels in output subscripts: {duplicates}")

    # Build the result shape
    result_shape = [label_dim_order[label] for label in output_labels]

    # Generate a new name for the result tensor
    result_name = generate_name()

    # Return new TensorProxy
    return TensorProxy(name=result_name, shape=result_shape, dtype=result_dtype)


if __name__ == "__main__":

    from Einsum_Compiler.Shape_Dtype import NOCC, NVIR, NMO, F32, F64

    # Define sample tensors
    tensor_a = TensorProxy(name="A", shape=[NOCC, NVIR], dtype=F32)
    tensor_b = TensorProxy(name="B", shape=[NVIR, NMO], dtype=F64)

    # Perform a simple einsum operation
    result = einsum("ij,jk->ik", tensor_a, tensor_b)
    print(f"Result of einsum('ij,jk->ik', tensor_a, tensor_b): {result}")

    # Additional tests
    # Test 1: Outer product
    tensor_c = TensorProxy(name="C", shape=[NOCC], dtype=F32)
    tensor_d = TensorProxy(name="D", shape=[NMO], dtype=F64)
    result_outer = einsum("i,j->ij", tensor_c, tensor_d)
    print(f"Result of einsum('i,j->ij', tensor_c, tensor_d): {result_outer}")

    # # Test 2: Summation over diagonal elements
    # tensor_e = TensorProxy(name="E", shape=[NOCC, NOCC], dtype=F32)
    # result_diag = einsum("ii->", tensor_e)
    # print(f"Result of einsum('ii->', tensor_e): {result_diag}")
