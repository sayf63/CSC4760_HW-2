#include <Kokkos_Core.hpp>
#include <iostream>
#include <limits>
#include <cstdlib>

int main(int argc, char* argv[]) {
	// Start Kokkos runtime.
	Kokkos::initialize(argc, argv);
	{
		// Default vector length.
		int n = 20;
		// Read optional n from command line.
		if (argc > 1) {
			n = std::atoi(argv[1]);
		}

		// Validate input.
		if (n <= 0) {
			std::cerr << "n must be a positive integer.\n";
			Kokkos::finalize();
			return 1;
		}

		// Allocate 1D integer View.
		Kokkos::View<int*> values("values", n);

		// Fill vector in parallel.
		Kokkos::parallel_for(
			"fill_values",
			Kokkos::RangePolicy<>(0, n),
			KOKKOS_LAMBDA(const int i) {
				// Example sequence value.
				values(i) = 3 * i + 1;
			});

		// Reduce to the maximum value in parallel.
		int max_value = std::numeric_limits<int>::lowest();
		Kokkos::parallel_reduce(
			"max_reduce",
			Kokkos::RangePolicy<>(0, n),
			KOKKOS_LAMBDA(const int i, int& local_max) {
				// Keep larger value.
				if (values(i) > local_max) {
					local_max = values(i);
				}
			},
			Kokkos::Max<int>(max_value));

		// Print reduction result.
		std::cout << "Maximum element in View: " << max_value << '\n';
	}
	// Shut down Kokkos runtime.
	Kokkos::finalize();
	return 0;
}
