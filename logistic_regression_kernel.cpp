#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "seal/seal.h"
#include "helper.h"

using namespace std;
using namespace seal;

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

// Matrix Transpose
vector<vector<double>> transpose_matrix(vector<vector<double>> input_matrix)
{

    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();
    vector<vector<double>> transposed(colSize, vector<double>(rowSize));

    for (int i = 0; i < rowSize; i++)
    {
        for (int j = 0; j < colSize; j++)
        {
            transposed[j][i] = input_matrix[i][j];
        }
    }

    return transposed;
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

// Standard Scaler
vector<vector<double>> standard_scaler(vector<vector<double>> input_matrix)
{
    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();
    vector<vector<double>> result_matrix(rowSize, vector<double>(colSize));

    // Optimization: Get Means and Standard Devs first then do the scaling
    // first pass: get means and standard devs
    vector<double> means_vec(colSize);
    vector<double> stdev_vec(colSize);
    for (int i = 0; i < colSize; i++)
    {
        vector<double> column(rowSize);
        for (int j = 0; j < rowSize; j++)
        {
            // cout << input_matrix[j][i] << ", ";
            column[j] = input_matrix[j][i];
            // cout << column[j] << ", ";
        }

        means_vec[i] = getMean(column);
        stdev_vec[i] = getStandardDev(column, means_vec[i]);
        // cout << "MEAN at i = " << i << ":\t" << means_vec[i] << endl;
        // cout << "STDV at i = " << i << ":\t" << stdev_vec[i] << endl;
    }

    // second pass: scale
    for (int i = 0; i < rowSize; i++)
    {
        for (int j = 0; j < colSize; j++)
        {
            result_matrix[i][j] = (input_matrix[i][j] - means_vec[j]) / stdev_vec[j];
            // cout << "RESULT at i = " << i << ":\t" << result_matrix[i][j] << endl;
        }
    }

    return result_matrix;
}

// Min Max Scaler
vector<vector<double>> minmax_scaler(vector<vector<double>> input_matrix)
{
    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();
    vector<vector<double>> result_matrix(rowSize, vector<double>(colSize));

    // Optimization: Get Means and Standard Devs first then do the scaling
    // first pass: get means and standard devs
    vector<double> means_vec(colSize);
    vector<double> stdev_vec(colSize);
    for (int i = 0; i < colSize; i++)
    {
        vector<double> column(rowSize);
        for (int j = 0; j < rowSize; j++)
        {
            // cout << input_matrix[j][i] << ", ";
            column[j] = input_matrix[j][i];
            // cout << column[j] << ", ";
        }

        means_vec[i] = getMean(column);
        stdev_vec[i] = getStandardDev(column, means_vec[i]);
        // cout << "MEAN at i = " << i << ":\t" << means_vec[i] << endl;
        // cout << "STDV at i = " << i << ":\t" << stdev_vec[i] << endl;
    }

    // second pass: scale
    for (int i = 0; i < rowSize; i++)
    {
        for (int j = 0; j < colSize; j++)
        {
            result_matrix[i][j] = (input_matrix[i][j] - means_vec[j]) / stdev_vec[j];
            // cout << "RESULT at i = " << i << ":\t" << result_matrix[i][j] << endl;
        }
    }

    return result_matrix;
}
int main()
{
    int poly_modulus_degree = 16384;
    EncryptionParameters params(scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    cout << "MAX BIT COUNT: " << CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
    params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 35, 35, 35, 35, 35, 35, 35, 60}));
    SEALContext context(params);

    print_parameters(context);

    cout << "Print the modulus switching chain" << endl;

    // Print the key level parameter info
    auto context_data = context.key_context_data();
    cout << "\tLevel (chain index): " << context_data->chain_index() << endl;
    cout << "\tparms_id: " << context_data->parms_id() << endl;
    cout << "\tcoeff_modulus primes: ";
    cout << hex;
    for (const auto &prime : context_data->parms().coeff_modulus())
    {
        cout << prime.value() << " ";
    }
    cout << dec << endl;
    cout << "\\" << endl;
    cout << " \\-->";

    // Iterate over the remaining levels
    context_data = context.first_context_data();
    while (context_data)
    {
        cout << " Level (chain index): " << context_data->chain_index();
        if (context_data->parms_id() == context.first_parms_id())
        {
            cout << " ...... first_context_data()" << endl;
        }
        else if (context_data->parms_id() == context.last_parms_id())
        {
            cout << " ...... last_context_data()" << endl;
        }
        else
        {
            cout << endl;
        }
        cout << "      parms_id: " << context_data->parms_id() << endl;
        cout << "      coeff_modulus primes: ";
        cout << hex;
        for (const auto &prime : context_data->parms().coeff_modulus())
        {
            cout << prime.value() << " ";
        }
        cout << dec << endl;
        cout << "\\" << endl;
        cout << " \\-->";

        // Step forward in the chain.
        context_data = context_data->next_context_data();
    }

    cout << "End of chain reached\n"
         << endl;

    // Generate keys, encryptor, decryptor and evaluator
    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();
    PublicKey pk;
    keygen.create_public_key(pk);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    Encryptor encryptor(context, pk);
    Evaluator evaluator(context);
    Decryptor decryptor(context, sk);

    // Create CKKS encoder
    CKKSEncoder ckks_encoder(context);

    // Create Scale
    double scale = pow(2.0, 35);

    // read file
    string filename = "pulsar_stars.csv";
    vector<vector<string>> s_matrix = CSVtoMatrix(filename);
    vector<vector<double>> f_matrix = stringTodoubleMatrix(s_matrix);

    // Init features, labels and weights
    // Init features (rows of f_matrix , cols of f_matrix - 1)
    int rows = f_matrix.size();
    rows = 100;
    cout << "\nNumber of rows  = " << rows << endl;
    int cols = f_matrix[0].size() - 1;
    cout << "\nNumber of cols  = " << cols << endl;

    vector<vector<double>> features(rows, vector<double>(cols));
    // Init labels (rows of f_matrix)
    vector<double> labels(rows);
    // Init weight vector with zeros (cols of features)
    vector<double> beta(rows);

    // Fill the features matrix and labels vector
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            features[i][j] = f_matrix[i][j];
        }
        labels[i] = f_matrix[i][cols];
    }

    // Fill the weights with random numbers (from 1 - 2)
    for (int i = 0; i < rows; i++)
    {
        beta[i] = RandomFloat(-1, 1) + 0.00000001;
    }

    vector<vector<double>> standard_features = standard_scaler(features);

    double lambda = 0.01;
    double poly_deg = 3 + 1;
    // double poly_deg = 7 + 1;
    
    // seperate features into two parts
    int col_A = 3;
    int col_B = cols - col_A;

    vector<vector<double>> kernel(rows, vector<double>(rows));
    vector<vector<double>> kernel_A(rows, vector<double>(rows));
    vector<vector<double>> kernel_B(rows, vector<double>(rows));

    // init to 0
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            kernel_A[i][j] = 0;
            kernel_B[i][j] = 0;
        }
    }

    // -------- LINEAR KERNEL --------
    cout << " -------- LINEAR KERNEL -------- " << endl;
    // calculate kernel_A
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            for(int k = 0; k < col_A; k++) { 
                kernel_A[i][j] += 
                    standard_features[i][k] * standard_features[j][k] + 0.00000001;
            }
        }
    }

    // calculate kernel_B
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            for(int k = 0; k < col_B; k++) { 
                kernel_B[i][j] += 
                    standard_features[i][col_A + k] * standard_features[j][col_A + k] + 0.00000001;
            }
        }
    }

    // Combine two kernels together
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            kernel[i][j] = kernel_A[i][j] + kernel_B[i][j];
            // cout << kernel[i][j] << " ";
        }
        // cout << endl;
    }

    // -------- Polynomial KERNEL --------
    // cout << " -------- Polynomial KERNEL -------- " << endl;

    // -------- RBF KERNEL --------
    // cout << " -------- RBF KERNEL -------- " << endl;


    vector<double> coeffs = {0.50101, 0.12669, -0.00005, -0.0009};
    // vector<double> coeffs = {0.50054, 0.19688, -0.00014, -0.00544, 0.000005, 0.000075, -0.00000004, -0.0000003};
    double learning_rate = 0.0001;
    int iter_times = 15;
    
    // Calculate gradient descents in the plaintext domain
    vector<double> delta_beta(rows, 0.0);
    vector<double> reg_term(rows, 0.0);

    for(int iter = 0; iter < iter_times; iter++) {
        for(int i = 0; i < rows; i++) {
            double b_k = 0.0;
            double tmp = 0.0;

            for(int j = 0; j < rows; j++) {
                b_k += beta[j] * kernel[i][j];
            }
            cout << b_k << endl;

            for(int j = 0; j < poly_deg; j++) {
                tmp = coeffs[j] * pow(-1 * labels[i], j + 1) / rows;
                tmp = tmp * pow(b_k, j);
                for(int k = 0; k < rows; k++) {
                    delta_beta[k] += tmp * kernel[i][k];
                }
            }
        }

        reg_term = linear_transformation(kernel, beta);
        for(int i = 0; i < rows; i++) {
            beta[i] = beta[i] - learning_rate * (reg_term[i] * 2 * lambda / rows + delta_beta[i]);
        }

        cout << "iter " << iter << endl;
        for(int i = 0; i < rows; i++) {
            cout << beta[i] << " ";
        }
        cout << endl;

    }


    // Calculate gradient descents in the encrypted domain

    /*

    // --------------- ENCODING ----------------
    cout << "ENCODING......\n";
    vector<Plaintext> kernel_A_plain(rows), kernel_B_plain(rows);

    for (int i = 0; i < rows; i++) {
        ckks_encoder.encode(kernel_A[i], scale, kernel_A_plain[i]);
        ckks_encoder.encode(kernel_B[i], scale, kernel_B_plain[i]);
    }

    // --------------- ENCRYPTNG ------------------
    cout << "ENCRYPTING......\n";
    vector<Ciphertext> features_A_cipher(rows), features_B_cipher(rows);
    for (int i = 0; i < rows; i++) {
        encryptor.encrypt(features_A_plain[i], features_A_cipher[i]);
        encryptor.encrypt(features_B_plain[i], features_B_cipher[i]);
    }
   
    // --------------- CALCULATTNG ------------------
    cout << "CALCULATING......\n";
    vector<Ciphertext> features_cipher(rows);  // x
    for(int i = 0; i < rows; i++) {
        evaluator.rotate_vector(features_B_cipher[i], -col_A, gal_keys, features_cipher[i]);
        evaluator.add_inplace(features_cipher[i], features_A_cipher[i]);
    }

    double one = 1;
    Plaintext one_plain;
    Ciphertext one_cipher;
    ckks_encoder.encode(one, scale, one_plain);
    encryptor.encrypt(one_plain, one_cipher);

    double alpha;
    Plaintext alpha_plain, weights_plain;
    Ciphertext x_cipher, weights_features_cipher, delta_w_all_cipher;
    vector<Ciphertext> delta_w_cipher(rows);
    vector<Ciphertext> wx_powers_cipher(poly_deg);
    // used when decoding
    Plaintext delta_w_plain;
    vector<double> delta_w_decode(cols);

    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_diff;
    time_start = chrono::high_resolution_clock::now();

    for(int iter = 0; iter < iter_times; iter++) {
        cout << "iter " << iter << endl;
        ckks_encoder.encode(weights, scale, weights_plain);
        for(int i = 0; i < rows; i++) {
            x_cipher = features_cipher[i];
            evaluator.multiply_plain(x_cipher, weights_plain, weights_features_cipher);
            evaluator.rescale_to_next_inplace(weights_features_cipher);

            compute_all_powers(weights_features_cipher, poly_deg, evaluator, relin_keys, wx_powers_cipher);
            wx_powers_cipher[0] = one_cipher;
            // cout << j << " " << wx_powers_cipher[j].parms_id() << endl;

            for(int j = 0; j < poly_deg; j++) {
                alpha = coeffs[j] * pow(-1 * labels[i], j + 1) / rows;
                ckks_encoder.encode(alpha, scale, alpha_plain);

                evaluator.mod_switch_to_inplace(alpha_plain, wx_powers_cipher[j].parms_id());
                evaluator.multiply_plain_inplace(wx_powers_cipher[j], alpha_plain);
                evaluator.rescale_to_next_inplace(wx_powers_cipher[j]);

                evaluator.mod_switch_to_inplace(x_cipher, wx_powers_cipher[j].parms_id());
                evaluator.multiply_inplace(wx_powers_cipher[j], x_cipher);
            }

            int last_id = wx_powers_cipher.size() - 1;
            parms_id_type last_parms_id = wx_powers_cipher[last_id].parms_id();
            double last_scale = pow(2, (int)log2(wx_powers_cipher[last_id].scale()));
            cout << last_parms_id << endl;

            for(int j = 0; j < poly_deg; j++) {
                evaluator.mod_switch_to_inplace(wx_powers_cipher[j], last_parms_id);
                wx_powers_cipher[j].scale() = last_scale;
            }

            evaluator.add_many(wx_powers_cipher, delta_w_cipher[i]);
        }

        evaluator.add_many(delta_w_cipher, delta_w_all_cipher);


        // Test
        decryptor.decrypt(delta_w_all_cipher, delta_w_plain);
        ckks_encoder.decode(delta_w_plain, delta_w_decode);

        for(int i = 0; i < cols; i++) {
            weights[i] = weights[i] - learning_rate * delta_w_decode[i];
            cout << weights[i] << " ";
        }
        cout << endl;
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << rows << " total execution time :\t" << time_diff.count() << " microseconds" << endl;

    // acuracy
    double acc_1 = 0.0, acc_2 = 0.0;
    for(int i = 0; i < rows; i++) {
        double tmp_1 = 0.0, tmp_2 = 0.0;
        for(int j = 0; j < cols; j++) {
            tmp_1 += w[j] * standard_features[i][j];
            tmp_2 += weights[j] * standard_features[i][j];
        }
        if(tmp_1 >= 0) {
            tmp_1 = 1;
        } else {
            tmp_1 = -1;
        }

        if(tmp_2 >= 0) {
            tmp_2 = 1;
        } else {
            tmp_2 = -1;
        }

        if(tmp_1 == labels[i]) acc_1 += 1;
        if(tmp_2 == labels[i]) acc_2 += 1;
    }
    cout << "acc 1 " << acc_1 / rows << endl;
    cout << "acc 2 " << acc_2 / rows << endl;

    */

    return 0;
}
