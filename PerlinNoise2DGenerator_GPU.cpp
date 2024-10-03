#include <iostream>
#include <random> // For std::random_device and std::mt19937
#include <CL/sycl.hpp>
#include "PerlinNoise2DGenerator_GPU.h"


PerlinNoise2DGenerator_GPU::PerlinNoise2DGenerator_GPU(int widthX, int heightY, int frequency, bool useGPU, int seedIn, bool seamlessVertically, bool seamlessHorizontally)
{
	f = frequency;

	// Recalculate width and height of array
	int newWidthX = (widthX - 1) % (f + 1) == 0 ? widthX : widthX + f + 1 - ((widthX - 1) % (f + 1));
	int newHeightY = (heightY - 1) % (f + 1) == 0 ? heightY : heightY + f + 1 - ((heightY - 1) % (f + 1));

	// Save current width and height to variables
	Width = newWidthX;
	Height = newHeightY;

	// Setting seed in the object
	if (seedIn == 0)
	{
		std::random_device rd;
		seed = rd();
	}
	else seed = seedIn;


	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// choose device to use 
	auto deviceToUse = useGPU ? sycl::gpu_selector_v : sycl::cpu_selector_v;

	// create a queue object to get access to the device  
	sycl::queue q(deviceToUse);


	// Declare buffer to store initial noise grid
	sycl::buffer<Point, 2> initialNoiseGrid_buffer(sycl::range<2>((newHeightY / (f + 1)) + 2, (newWidthX / (f + 1)) + 2));



	q.submit([&](sycl::handler& h)
		{
			// Setting object variables as local
			int seed = this->getSeed();

			auto initNoise_acc = initialNoiseGrid_buffer.get_access<sycl::access::mode::write>(h);

			auto out = sycl::stream(1024, 128, h);

			h.parallel_for(sycl::range<2>((newHeightY / (f + 1)) + 2, (newWidthX / (f + 1)) + 2), [=](sycl::item<2> i)
				{

					// All possible vectors for initial noise
					Point possibleVectors[]{ {1.0,1.0}, {-1.0,1.0}, {1.0,-1.0}, {-1.0,-1.0}, {std::sqrt(2),0}, {0,std::sqrt(2)}, {-std::sqrt(2),0}, {0,-std::sqrt(2)} };  //1.4142

					// Random selection of Point values based on seed value
					std::mt19937 firstRandomization(seed * (i.get_id(0) + 1) + i.get_id(1));
					std::mt19937 secondRandomization(firstRandomization() * (i.get_id(1) + 1) + i.get_id(0));
					std::mt19937 randChoise(secondRandomization());

					initNoise_acc[i.get_id(0)][i.get_id(1)] = possibleVectors[randChoise() % std::size(possibleVectors)];
				});

		}).wait();


		{
			sycl::host_accessor initNoise_acc(initialNoiseGrid_buffer);
			
			// Make the noise seamless along Y axis
			if (seamlessVertically)
			{
				for (int i = 0; i <= (newHeightY / (f + 1)); ++i)
				{
					initNoise_acc[i][(newWidthX / (f + 1))] = initNoise_acc[i][0];
				}
			}
			// Make the noise seamless along X axis
			if (seamlessHorizontally)
			{
				for (int i = 0; i <= (newWidthX / (f + 1)); ++i)
				{
					initNoise_acc[(newHeightY / (f + 1))][i] = initNoise_acc[0][i];
				}
			}
		}


	// Set size of blackAndWhiteNoise2DArray 
	blackAndWhiteNoiseArray.resize(newWidthX * newHeightY);

	// Create buffer for result array 
	sycl::buffer<int, 1>  blackAndWhiteNoiseArray_buffer(blackAndWhiteNoiseArray.data(), sycl::range<1>(blackAndWhiteNoiseArray.size()));


	q.submit([&](sycl::handler& h)
		{
			// Setting object variables as local
			int f = this->f;

			auto initialNoiseGrid_acc = initialNoiseGrid_buffer.get_access<sycl::access::mode::read>(h);

			// Create accessor for blackAndWhiteNoise2DArray_buffer
			auto outputArray_accessor = blackAndWhiteNoiseArray_buffer.get_access<sycl::access::mode::write>(h);

			auto out = sycl::stream(1024, 128, h);
			

			h.parallel_for(sycl::range<2>(newHeightY, newWidthX), [=](sycl::item<2> item)
				{
					// Set secure access to a shared variable
					sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> element(outputArray_accessor[item.get_linear_id()]);

					

					// set x and y
					int x = item.get_id(0);
					int y = item.get_id(1);

					
					// Creating 4 vectors from 4 nearest base noise dots to pixel position
					Point vectorLT;
					vectorLT.x = (x % (f + 1)) * (1 / float(f + 1));
					vectorLT.y = (y % (f + 1)) * (1 / float(f + 1));

					Point vectorRT;
					vectorRT.x = ((x % (f + 1)) - f - 1) * (1 / float(f + 1));
					vectorRT.y = (y % (f + 1)) * (1 / float(f + 1));

					Point vectorLB;
					vectorLB.x = (x % (f + 1)) * (1 / float(f + 1));
					vectorLB.y = ((y % (f + 1)) - f - 1) * (1 / float(f + 1));

					Point vectorRB;
					vectorRB.x = ((x % (f + 1)) - f - 1) * (1 / float(f + 1));
					vectorRB.y = ((y % (f + 1)) - f - 1) * (1 / float(f + 1));

					

					// Doing dot product of 4 nearest base noise dots and previous 4 vectors
					float LTDp = ( vectorLT.x * initialNoiseGrid_acc[int(x / (f + 1))][int(y / (f + 1))].x + vectorLT.y * initialNoiseGrid_acc[int(x / (f + 1))][int(y / (f + 1))].y );
					float RTDp = ( vectorRT.x * initialNoiseGrid_acc[int(x / (f + 1)) + 1][int(y / (f + 1))].x + vectorRT.y * initialNoiseGrid_acc[int(x / (f + 1)) + 1][int(y / (f + 1))].y);
					float LBDp = ( vectorLB.x * initialNoiseGrid_acc[int(x / (f + 1))][int(y / (f + 1)) + 1].x + vectorLB.y * initialNoiseGrid_acc[int(x / (f + 1))][int(y / (f + 1)) + 1].y);
					float RBDp = ( vectorRB.x * initialNoiseGrid_acc[int(x / (f + 1)) + 1][int(y / (f + 1)) + 1].x + vectorRB.y * initialNoiseGrid_acc[int(x / (f + 1)) + 1][int(y / (f + 1)) + 1].y);

					// Precalculate values of fade for x and y
					float fadeX = 6 * std::pow((x % (f + 1)) * (1 / float(f + 1)), 5) - 15 * std::pow((x % (f + 1)) * (1 / float(f + 1)), 4) + 10 * std::pow((x % (f + 1)) * (1 / float(f + 1)), 3);
					float fadeY = 6 * std::pow((y % (f + 1)) * (1 / float(f + 1)), 5) - 15 * std::pow((y % (f + 1)) * (1 / float(f + 1)), 4) + 10 * std::pow((y % (f + 1)) * (1 / float(f + 1)), 3);

					// Finding interpolation with fade for "top" two dot products
					float UpLerp = LTDp + fadeX * (RTDp - LTDp);
					// Finding interpolation with fade for "bottom" two dot products
					float DownLerp = LBDp + fadeX * (RBDp - LBDp);
					// Finding result value for Perling noise in position x y
					float finalLerp = UpLerp + fadeY * (DownLerp - UpLerp);
					
					// Convert finalLerp value to black and white value
					// Map the value to [0, 255]
					int grayValue = static_cast<int>((finalLerp + 1.0f) / 2.0f * 255);

					// Save grayValue to blackAndWhiteNoise2DArray
					element.store(grayValue);
	
				});

		}).wait();

}

int PerlinNoise2DGenerator_GPU::getColourAtPoint(int x, int y)
{
	return blackAndWhiteNoiseArray[y * Width + x];
}
