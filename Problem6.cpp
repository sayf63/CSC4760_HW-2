#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

int main(int argc, char* argv[]) {
	// Start Kokkos runtime.
	Kokkos::initialize(argc, argv);
	{
		// Default benchmark settings.
		int num_rows = 2000;
		int num_cols = 2000;
		int repeats = 10;

		// Read optional command-line overrides.
		if (argc > 1) {
			num_rows = std::atoi(argv[1]);
		}
		if (argc > 2) {
			num_cols = std::atoi(argv[2]);
		}
		if (argc > 3) {
			repeats = std::atoi(argv[3]);
		}

		// Validate benchmark inputs.
		if (num_rows <= 0 || num_cols <= 0 || repeats <= 0) {
			std::cerr << "Usage: " << argv[0]
				<< " [num_rows > 0] [num_cols > 0] [repeats > 0]\n";
			Kokkos::finalize();
			return 1;
		}

		// Type aliases for matrix and row-sum vector.
		using View2D = Kokkos::View<double**>;
		using View1D = Kokkos::View<double*>;

		// Allocate matrix and output row-sum buffer.
		View2D matrix("matrix", num_rows, num_cols);
		View1D parallel_row_sums("parallel_row_sums", num_rows);

		// Initialize matrix values in parallel.
		Kokkos::parallel_for(
			"initialize_matrix",
			Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_rows, num_cols}),
			KOKKOS_LAMBDA(const int i, const int j) {
				// Small repeating pattern to avoid trivial constants.
				matrix(i, j) = 1.0 + static_cast<double>((i + j) % 7);
			});

		// Ensure initialization is complete.
		Kokkos::fence();

		// Host copy for serial baseline loop.
		auto matrix_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix);
		double serial_checksum = 0.0;
		double parallel_checksum = 0.0;

		// Time serial row-sum loop.
		Kokkos::Timer serial_timer;
		for (int r = 0; r < repeats; ++r) {
			double pass_checksum = 0.0;
			for (int i = 0; i < num_rows; ++i) {
				double row_sum = 0.0;
				for (int j = 0; j < num_cols; ++j) {
					row_sum += matrix_host(i, j);
				}
				// Accumulate row result into pass checksum.
				pass_checksum += row_sum;
			}
			// Keep checksum from the latest pass.
			serial_checksum = pass_checksum;
		}
		double serial_seconds = serial_timer.seconds();

		// Time parallel row-sum kernel.
		Kokkos::Timer parallel_timer;
		for (int r = 0; r < repeats; ++r) {
			Kokkos::parallel_for(
				"row_sums_parallel",
				Kokkos::RangePolicy<>(0, num_rows),
				KOKKOS_LAMBDA(const int i) {
					double row_sum = 0.0;
					for (int j = 0; j < num_cols; ++j) {
						row_sum += matrix(i, j);
					}
					// Store one sum per row.
					parallel_row_sums(i) = row_sum;
				});

			// Ensure each timed pass completes.
			Kokkos::fence();
		}
		double parallel_seconds = parallel_timer.seconds();

		// Reduce row sums to one checksum value.
		Kokkos::parallel_reduce(
			"parallel_checksum",
			Kokkos::RangePolicy<>(0, num_rows),
			KOKKOS_LAMBDA(const int i, double& update) {
				update += parallel_row_sums(i);
			},
			parallel_checksum);

		// Ensure reduction is complete.
		Kokkos::fence();

		// Compare outputs and timing.
		double diff = std::fabs(serial_checksum - parallel_checksum);
		double speedup = serial_seconds / parallel_seconds;

		// Print benchmark summary.
		std::cout << "Rows: " << num_rows << ", Cols: " << num_cols
			<< ", Repeats: " << repeats << '\n';
		std::cout << "Serial checksum:   " << serial_checksum << '\n';
		std::cout << "Parallel checksum: " << parallel_checksum << '\n';
		std::cout << "Checksum diff:     " << diff << '\n';
		std::cout << "Serial time (s):   " << serial_seconds << '\n';
		std::cout << "Parallel time (s): " << parallel_seconds << '\n';
		std::cout << "Speedup:           " << speedup << "x\n";
	}
	// Shut down Kokkos runtime.
	Kokkos::finalize();
	return 0;
}
