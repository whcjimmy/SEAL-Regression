#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <vector>
#include <valarray>
#include "util.cpp"

using namespace std;

void test_accuracy(vector<double> weights, vector<vector<double>> testing_x, vector<double> testing_y) {
    double accuracy = 0.0, tmp;
    int testing_rows = testing_x.size();
    for(int i = 0; i < testing_rows; i++) {
        tmp = vector_dot_product(weights, testing_x[i]);
        if(tmp >= 0) {
            tmp = 1;
        } else {
            tmp = -1;
        }

        if(tmp == testing_y[i]) accuracy += 1;
    }
    cout << "accuracy " << accuracy / testing_rows << endl;
}


int main()
{
    // read file
    // string filename = "pulsar_stars.csv";
    // string filename = "./python/datasets/Heart-Disease-Machine-Learning/data.csv";
    string filename = "./python/datasets/breast_cancer/data.csv";
    vector<vector<string>> s_matrix = CSVtoMatrix(filename);
    vector<vector<double>> f_matrix = stringTodoubleMatrix(s_matrix);
    // random_shuffle(f_matrix.begin(), f_matrix.end());

    // Init features, labels and weights
    int total_rows = f_matrix.size();
    cout << "\nNumber of rows  = " << total_rows << endl;
    int total_cols = f_matrix[0].size() - 1;
    cout << "\nNumber of cols  = " << total_cols << endl;

    vector<vector<double>> features(total_rows, vector<double>(total_cols));
    // Init labels (rows of f_matrix)
    vector<double> labels(total_rows);

    int training_rows = 364;
    int testing_rows = total_rows - training_rows;
    vector<vector<double>> training_x(training_rows, vector<double>(total_cols));
    vector<vector<double>> testing_x(testing_rows, vector<double>(total_cols));
    vector<double> training_y(training_rows);
    vector<double> testing_y(testing_rows);

    // Fill the features matrix and labels vector
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_cols; j++) {
            if(i < training_rows) {
                training_x[i][j] = f_matrix[i][j];
            } else {
                testing_x[i - training_rows][j] = f_matrix[i][j];
            }
        }
        if(i < training_rows) {
            training_y[i] = f_matrix[i][total_cols];
        } else {
            testing_y[i - training_rows] = f_matrix[i][total_cols];
        }
    }

    // Init weights
    // Init weight vector with zeros (cols of features)
    vector<double> weights(total_cols);
    for (int i = 0; i < total_cols; i++) {
        weights[i] = 0;
        // weights[i] = RandomFloat(-0.1, 0.1);
    }

    // Polynomial Simulated Sigmoid Function
    double poly_deg = 3;
    // double poly_deg = 7;

    vector<double> coeffs = {0.50081, 0.08937, -0.00001, -0.00297};
    // vector<double> coeffs = {0.50054, 0.19688, -0.00014, -0.00544, 0.000005, 0.000075, -0.00000004, -0.0000003};


    // Parameters Settings
    int col_A = 15;
    int col_B = total_cols - col_A;

    double learning_rate = 1;
    int iter_times = 30;

    // Calculate gradient descents in the plaintext domain
    vector<double> delta_w(total_cols, 0.0);
    double w_x, tmp;
    for(int iter = 0; iter < iter_times; iter++) {
        cout << "iter " << iter << endl;

        fill(delta_w.begin(), delta_w.end(), 0);

        for(int i = 0; i < training_rows; i++) {
            w_x = vector_dot_product(weights, training_x[i]);

            tmp = 0.0;
            for(int j = 0; j <= poly_deg; j++) {
                tmp += coeffs[j] * pow(-1 * training_y[i], j + 1) * pow(w_x, j);
            }

            for(int j = 0; j < total_cols; j++) {
                delta_w[j] += tmp * training_x[i][j];
            }
        }

        for(int i = 0; i < total_cols; i++) {
            weights[i] = weights[i] - learning_rate * delta_w[i] / training_rows;
        }

        // Show results
        for(int i = 0; i < 10; i++) {
            cout << weights[i] << " ";
        }
        cout << endl;

        test_accuracy(weights, training_x, training_y);
        test_accuracy(weights, testing_x, testing_y);
    }

    return 0;
}
