#include"Layer.hpp"
#include<memory>
#include<iostream>
#include<vector>

using namespace std;

void Conv::Init()
{
	cout << "Conv Init" << endl;
}

void Fc::Init()
{
	cout << "Fc Init" << endl;
}

void Pool::Init()
{
	cout << "Pool Init" << endl;
}

void Relu::Init()
{
	cout << "Relu Init" << endl;
}