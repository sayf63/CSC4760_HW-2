#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
	// Start Kokkos runtime.
	Kokkos::initialize(argc, argv);
	{
		// Create a 1D View of 10 doubles.
		Kokkos::View<double*> values("values_view", 10);
		// Print the View label.
		std::cout << "View label: " << values.label() << std::endl;
	}
	// Shut down Kokkos runtime.
	Kokkos::finalize();
	return 0;
}
