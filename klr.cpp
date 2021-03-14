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
    // string filename = "./python/datasets/breast_cancer/data.csv";
     string filename = "./python/datasets/make_circles.csv";
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

    int training_rows = 400;
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
    vector<double> weights(training_rows);
    for (int i = 0; i < training_rows; i++) {
        weights[i] = 0;
        // weights[i] = RandomFloat(-0.1, 0.1);
    }

    // Polynomial Simulated Sigmoid Function
    double poly_deg = 3;
    // double poly_deg = 7;

    vector<double> coeffs = {0.50091, 0.19832, -0.00018, -0.00447};
    // vector<double> coeffs = {0.50054, 0.19688, -0.00014, -0.00544, 0.000005, 0.000075, -0.00000004, -0.0000003};


    // Parameters Settings
    int col_A = 1;
    int col_B = total_cols - col_A;

    double learning_rate = 0.01;
    int iter_times = 20;
    double lambda = 0.1;
    double gamma  = 1;
    int poly_kernel_deg = 3;

    // KERNEL
    vector<vector<double>> training_kernel(training_rows, vector<double>(training_rows));
    vector<vector<double>> testing_kernel(testing_rows, vector<double>(training_rows));

    /*
    cout << "---------- LINEAR KERNEL ----------" <<endl;
    // Training Kernel
    for(int i = 0; i < training_rows; i++) {
        for(int j = 0; j < training_rows; j++) { 
            training_kernel[i][j] = vector_dot_product(training_x[i], training_x[j]);
        }
    }
    // Testing Kernel
    for(int i = 0; i < testing_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            testing_kernel[i][j] = vector_dot_product(testing_x[i], training_x[j]);
        }
    }
    */

    /*
    cout << "---------- POLYNOMIAL KERNEL ----------" <<endl;
    // Training Kernel
    for(int i = 0; i < training_rows; i++) {
        for(int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < total_cols; k++) {
                training_kernel[i][j] += training_x[i][k] * training_x[j][k];
            }
            training_kernel[i][j] = training_kernel[i][j] * gamma;
            training_kernel[i][j] = 1 + training_kernel[i][j];
            training_kernel[i][j] = pow(training_kernel[i][j], poly_kernel_deg);
        }
    }
    // Testing Kernel
    for(int i = 0; i < testing_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < total_cols; k++) { 
                testing_kernel[i][j] += testing_x[i][k] * training_x[j][k];
            }
            testing_kernel[i][j] = testing_kernel[i][j] * gamma;
            testing_kernel[i][j] = 1 + testing_kernel[i][j];
            testing_kernel[i][j] = pow(testing_kernel[i][j], poly_kernel_deg);
        }
    }
    */

    cout << "---------- RBF KERNEL ----------" <<endl;
    // Training Kernel
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < total_cols; k++) {
                training_kernel[i][j] += pow(training_x[i][k] - training_x[j][k], 2);
            }
            training_kernel[i][j] = -1 * training_kernel[i][j] * gamma;
            training_kernel[i][j] = 1 + training_kernel[i][j] + pow(training_kernel[i][j], 2) / 2;
        }
    }
    // Testing Kernel
    for(int i = 0; i < testing_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < total_cols; k++) { 
                testing_kernel[i][j] += pow(testing_x[i][k] - training_x[j][k], 2);
            }
            testing_kernel[i][j] = -1 * testing_kernel[i][j] * gamma;
            testing_kernel[i][j] = 1 + testing_kernel[i][j] + pow(testing_kernel[i][j], 2) / 2;
        }
    }


    // Calculate gradient descents in the plaintext domain
    vector<double> delta_w(training_rows, 0.0);
    vector<double> l2_reg(training_rows, 0.0);
    double w_x, tmp;
    for(int iter = 0; iter < iter_times; iter++) {
        cout << "iter " << iter << endl;

        fill(delta_w.begin(), delta_w.end(), 0);

        for(int i = 0; i < training_rows; i++) {
            w_x = vector_dot_product(weights, training_kernel[i]);

            tmp = 0.0;
            for(int j = 0; j <= poly_deg; j++) {
                tmp += coeffs[j] * pow(-1 * training_y[i], j + 1) * pow(w_x, j);
            }

            for(int j = 0; j < training_rows; j++) {
                delta_w[j] += tmp * training_kernel[i][j];
            }
        }

        l2_reg = linear_transformation(training_kernel, weights);
        tmp = 2 * lambda / training_rows;

        for(int i = 0; i < training_rows; i++) {
            weights[i] = weights[i] - learning_rate * (l2_reg[i] * tmp + delta_w[i] / training_rows);
        }

        // Show results
        for(int i = 0; i < 10; i++) {
            cout << weights[i] << " ";
        }
        cout << endl;

        test_accuracy(weights, training_kernel, training_y);
        test_accuracy(weights, testing_kernel, testing_y);
    }

    return 0;
}
