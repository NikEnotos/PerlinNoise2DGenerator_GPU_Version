
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "PerlinNoise2DGenerator_GPU.h"
#include "SFML/Graphics.hpp"

using namespace sycl;

int main()
{
	// getting values for noise settings

	int width;
	int height;
	int friquency;

	int seed;
	bool verticallySeamless;
	bool horizontallySeamless;
	bool useGPU;

	
	std::cout << "Input next arguments:\n\n   Width(int) [Example: 1920]: "; 	std::cin >> width;
	std::cout << "\n   Height(int) [Example: 1080]: "; 	std::cin >> height;
	std::cout << "\n   Frequency(int) <range[>0]> [Example: 50]: "; 	std::cin >> friquency;
	std::cout << "\n   Seed(int) <0 for rendom> [Example: 42]: "; 	std::cin >> seed;
	std::cout << "\n   Vertically seamless(bool) [Example: 1 for 'true' / 0 for 'false']: "; 	std::cin >> verticallySeamless;
	std::cout << "\n   Horizontally seamless(bool) [Example: 1 for 'true' / 0 for 'false']: "; 	std::cin >> horizontallySeamless;
	std::cout << "\n   Use GPU(bool) [Example: 1 for 'true' / 0 for 'false']: "; 	std::cin >> useGPU;

	// determine available computing powers and show them

	auto deviceToUse = useGPU ? sycl::gpu_selector_v : sycl::cpu_selector_v;
	queue q(deviceToUse);
	
	std::cout << "\n   Device name: " << q.get_device().get_info<info::device::name>();
	std::cout << "\n   max_compute_units: " << q.get_device().get_info<info::device::max_compute_units>();


	{

		std::cout << "\n\n<<< GPU Perlin noise thread version >>> \n" << std::endl;

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Start the clock
		auto noise_Generation_start = std::chrono::high_resolution_clock::now();

		PerlinNoise2DGenerator_GPU noise(width, height, friquency, useGPU, seed, verticallySeamless, horizontallySeamless);



		// Create an SFML image to store the pixel data
		sf::Image image;
		image.create(noise.Width, noise.Height);

		//// Convert the values to black and white and set pixels in the image
		for (int y = 0; y < noise.Height; ++y) {
			for (int x = 0; x < noise.Width; ++x) {

				int grayValue = noise.getColourAtPoint(x, y);

				sf::Color pixelColor(grayValue, grayValue, grayValue);

				image.setPixel(x, y, pixelColor);

			}
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Stop the clock
		auto noise_Generation_end = std::chrono::high_resolution_clock::now();
		// Calculate the duration
		auto noise_Generation_duration = std::chrono::duration_cast<std::chrono::seconds>(noise_Generation_end - noise_Generation_start);
		// Print the duration
		std::cout << ">>> Noise Generation took: " << noise_Generation_duration.count() << " seconds" << std::endl;

		std::cout << ">>> Saving generated noise " << std::endl;



		std::string image_name = std::to_string(width) + "x" + std::to_string(height) + "_GPU-Parallel_PerlinNoise.png";

		if (!image.saveToFile(image_name))
		{
			std::cout << "[-] Image saving has failed :( " << std::endl << std::endl;
		}
		else
		{
			std::cout << "[+] Image saved successfully (" << image_name << ")" << std::endl << std::endl;
		}

	}

	std::system("pause");

	return 0;
}