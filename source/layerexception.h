#pragma once
#include <exception>
using namespace std;

class LayerException :
	public exception
{
public:
	LayerException(char const* const message) :
		exception(message)
	{
	}
};

