#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>

// 2D integer matrix View type.
using View2D = Kokkos::View<int**>;
// 1D integer vector View type.
using View1D = Kokkos::View<int*>;

// Adds vector B to every row of matrix A.
View2D add_vector_to_rows(const View2D& matrix, const View1D& vector) {
	// Get matrix and vector sizes.
	const int rows = static_cast<int>(matrix.extent(0));
	const int cols = static_cast<int>(matrix.extent(1));
	const int vec_size = static_cast<int>(vector.extent(0));

	// Make sure dimensions are compatible.
	if (cols != vec_size) {
		throw std::invalid_argument(
			"Dimension mismatch: matrix columns must equal vector length.");
	}

	// Allocate output matrix.
	View2D result("result", rows, cols);

	// Parallel loop over all (row, col) entries.
	Kokkos::parallel_for(
		"add_vector_to_rows",
		Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {rows, cols}),
		KOKKOS_LAMBDA(const int i, const int j) {
			// Add vector element j to matrix element (i, j).
			result(i, j) = matrix(i, j) + vector(j);
		});

	// Ensure kernel is complete before returning.
	Kokkos::fence();
	return result;
}

int main(int argc, char* argv[]) {
	// Start Kokkos runtime.
	Kokkos::initialize(argc, argv);
	{
		// Unused command-line args for this fixed test case.
		(void)argc;
		(void)argv;

		// Create test matrix A and vector B.
		View2D A("A", 3, 3);
		View1D B("B", 3);

		// Host mirrors for easy value assignment.
		auto A_host = Kokkos::create_mirror_view(A);
		auto B_host = Kokkos::create_mirror_view(B);

		// Fill matrix A values.
		A_host(0, 0) = 130; A_host(0, 1) = 147; A_host(0, 2) = 115;
		A_host(1, 0) = 224; A_host(1, 1) = 158; A_host(1, 2) = 187;
		A_host(2, 0) = 54;  A_host(2, 1) = 158; A_host(2, 2) = 120;

		// Fill vector B values.
		B_host(0) = 221; B_host(1) = 12; B_host(2) = 157;

		// Copy input data to device Views.
		Kokkos::deep_copy(A, A_host);
		Kokkos::deep_copy(B, B_host);

		// Compute C = A + B(row-wise broadcast).
		View2D C = add_vector_to_rows(A, B);
		// Copy result back to host for checking/printing.
		auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);

		// Expected answer for the provided test case.
		int expected[3][3] = {
			{351, 159, 272},
			{445, 170, 344},
			{275, 170, 277}
		};

		// Validate result against expected values.
		bool is_correct = true;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				if (C_host(i, j) != expected[i][j]) {
					is_correct = false;
				}
			}
		}

		// Print computed matrix C.
		std::cout << "A + B (B added to each row of A):\n";
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::cout << C_host(i, j) << (j + 1 == 3 ? '\n' : ' ');
			}
		}

		// Print correctness check result.
		std::cout << "Matches expected solution: "
			<< (is_correct ? "YES" : "NO") << '\n';
	}
	// Shut down Kokkos runtime.
	Kokkos::finalize();
	return 0;
}
