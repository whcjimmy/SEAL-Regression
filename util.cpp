#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <valarray>

using namespace std;

// Dot Product
double vector_dot_product(vector<double> vec_A, vector<double> vec_B)
{
    if (vec_A.size() != vec_B.size())
    {
        cerr << "Vector size mismatch" << endl;
        exit(1);
    }

    double result = 0;
    for (unsigned int i = 0; i < vec_A.size(); i++)
    {
        result += vec_A[i] * vec_B[i];
    }

    return result;
}


// Linear Transformation (or Matrix * Vector)
vector<double> linear_transformation(vector<vector<double>> input_matrix, vector<double> input_vec)
{

    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();

    if (colSize != input_vec.size())
    {
        cerr << "Matrix Vector sizes error" << endl;
        exit(EXIT_FAILURE);
    }

    vector<double> result_vec(rowSize);
    for (int i = 0; i < input_matrix.size(); i++)
    {
        result_vec[i] = vector_dot_product(input_matrix[i], input_vec);
    }

    return result_vec;
}


// CSV to string matrix converter
vector<vector<string>> CSVtoMatrix(string filename)
{
    vector<vector<string>> result_matrix;

    ifstream data(filename);
    string line;
    int line_count = 0;
    while (getline(data, line))
    {
        stringstream lineStream(line);
        string cell;
        vector<string> parsedRow;
        while (getline(lineStream, cell, ','))
        {
            parsedRow.push_back(cell);
        }
        result_matrix.push_back(parsedRow);
        // Skip first line since it has text instead of numbers
        /*
        if (line_count != 0)
        {
            result_matrix.push_back(parsedRow);
        }
        */
        line_count++;
    }
    return result_matrix;
}


// String matrix to double matrix converter
vector<vector<double>> stringTodoubleMatrix(vector<vector<string>> matrix)
{
    vector<vector<double>> result(matrix.size(), vector<double>(matrix[0].size()));
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            result[i][j] = ::atof(matrix[i][j].c_str());
        }
    }

    return result;
}


// Min Max Scaler
vector<vector<double>> minmax_scaler(vector<vector<double>> input_matrix)
{
    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();
    for(int j = 0; j < colSize; j++) {
        double max = -99999;
        double min = 99999;
        for(int i = 0; i < rowSize; i++) {
            if(input_matrix[i][j] > max) {
                max = input_matrix[i][j];
            }
            if(input_matrix[i][j] < min) {
                min = input_matrix[i][j];
            }
        }
        for(int i = 0; i < rowSize; i++) {
            input_matrix[i][j] = (input_matrix[i][j] - min) / (max - min);
        }
    }

    return input_matrix;
}


// Gets a random float between a and b
float RandomFloat(float a, float b)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

