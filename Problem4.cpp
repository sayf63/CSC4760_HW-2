#include <Kokkos_Core.hpp>
#include <cstdlib>

int main(int argc, char* argv[]) {
	// Start Kokkos runtime.
	Kokkos::initialize(argc, argv);
	{
		// Default size for the 4th dimension.
		int n = 10;
		// Read optional n from command line.
		if (argc > 1) {
			n = std::atoi(argv[1]);
		}

		// Validate dimension.
		if (n <= 0) {
			Kokkos::finalize();
			return 1;
		}

		// Allocate a 4D View with fixed first 3 dimensions.
		Kokkos::View<double****> values("values_4d", 5, 7, 12, n);
		// Silence unused variable warning.
		(void)values;
	}
	// Shut down Kokkos runtime.
	Kokkos::finalize();
	return 0;
}
