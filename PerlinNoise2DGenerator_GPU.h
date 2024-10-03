#pragma once
#include <vector>

struct Point
{
	float x;
	float y;
};

class PerlinNoise2DGenerator_GPU
{
public:

	PerlinNoise2DGenerator_GPU(int widthX, int heightY, int frequency, bool useGPU, int seedIn = 0, bool seamlessVertically = false, bool seamlessHorizontally = false);

	inline int getSeed() { return seed; }

	int getColourAtPoint(int x, int y);

	// TODO: Create getWidth() and getHeight()
	int Width;
	int Height;

	bool seamlessHorizontally, seamlessVertically;

private:

	int seed; // The seed is used to initialize the state of the generator and determines the sequence of random numbers it produces.
	int f; // Noise grid frequency

	std::vector<int> blackAndWhiteNoiseArray;

};