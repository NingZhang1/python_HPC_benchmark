import numpy as np
import torch
import BackEnd._config

BackEnd._config.disable_fftw()
import BackEnd._numpy
import BackEnd._torch

numpy_einsum_ik_jk_ijk = BackEnd._numpy.einsum_ik_jk_ijk
torch_einsum_ik_jk_ijk = BackEnd._torch.einsum_ik_jk_ijk

# For this test, we'll use the NumPy implementation
einsum_ik_jk_ijk = numpy_einsum_ik_jk_ijk


def test_einsum_implementations(device="cpu"):
    print(f"\nTesting on {device}")

    # Test case 1: Small random matrices
    A1 = np.random.rand(3, 4)
    B1 = np.random.rand(5, 4)

    # Test case 2: Larger random matrices
    A2 = np.random.rand(10, 20)
    B2 = np.random.rand(15, 20)

    # Test case 3: Matrices with specific values
    A3 = np.array([[1, 2], [3, 4]])
    B3 = np.array([[5, 6], [7, 8], [9, 10]])

    test_cases = [(A1, B1), (A2, B2), (A3, B3)]

    for i, (A, B) in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")

        # NumPy reference implementation
        np_result = np.einsum("ik,jk->ijk", A, B)

        # Test NumPy implementation
        numpy_result = numpy_einsum_ik_jk_ijk(A, B)
        numpy_correct = np.allclose(numpy_result, np_result)
        print(f"NumPy implementation correct: {numpy_correct}")

        # Test PyTorch implementation
        A_torch = torch.tensor(A, device=device)
        B_torch = torch.tensor(B, device=device)
        torch_result = torch_einsum_ik_jk_ijk(A_torch, B_torch)
        torch_correct = np.allclose(torch_result.cpu().numpy(), np_result)
        print(f"PyTorch implementation correct on {device}: {torch_correct}")

        # Test the chosen implementation (NumPy in this case)
        chosen_result = einsum_ik_jk_ijk(A, B)
        chosen_correct = np.allclose(chosen_result, np_result)
        print(f"Chosen implementation correct: {chosen_correct}")

        if not (numpy_correct and torch_correct and chosen_correct):
            print("Error detected. Printing details:")
            print(f"A shape: {A.shape}, B shape: {B.shape}")
            print(f"NumPy reference result:\n{np_result}")
            print(f"NumPy implementation result:\n{numpy_result}")
            print(
                f"PyTorch implementation result on {device}:\n{torch_result.cpu().numpy()}"
            )
            print(f"Chosen implementation result:\n{chosen_result}")


if __name__ == "__main__":
    test_einsum_implementations("cpu")
    if torch.cuda.is_available():
        test_einsum_implementations("cuda")
    else:
        print("\nCUDA is not available. GPU tests skipped.")
