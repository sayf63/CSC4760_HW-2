#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
	// Start Kokkos runtime.
	Kokkos::initialize(argc, argv);
	{
		// Default matrix size.
		int n = 4;
		int m = 5;
		// Read optional n and m from command line.
		if (argc >= 3) {
			n = std::atoi(argv[1]);
			m = std::atoi(argv[2]);
		}

		// Validate dimensions.
		if (n <= 0 || m <= 0) {
			std::cerr << "n and m must be positive integers.\n";
			Kokkos::finalize();
			return 1;
		}

		// Allocate a 2D View.
		Kokkos::View<long long**> values("values_2d", n, m);

		// Fill all entries in parallel.
		Kokkos::parallel_for(
			"fill_values",
			Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m}),
			KOKKOS_LAMBDA(const int i, const int j) {
				// Example formula for each element.
				values(i, j) = 1000LL * i * j;
			});
		// Wait for kernel completion.
		Kokkos::fence();

		// Copy data to host for printing.
		auto values_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);

		// Print matrix values.
		std::cout << "n x m View values (" << n << " x " << m << "):\n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				std::cout << values_host(i, j) << " ";
			}
			std::cout << '\n';
		}
	}
	// Shut down Kokkos runtime.
	Kokkos::finalize();
	return 0;
}
